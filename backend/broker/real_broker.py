"""KIS account/limit-order adapter. The experiment CLI never submits API orders.

Official request definitions checked against Korea Investment's public examples:
https://github.com/koreainvestment/open-trading-api/tree/main/examples_llm/domestic_stock
Use mode=paper for KIS VTS, mode=real for the real endpoint. enable_orders is
required for submission. Order responses acknowledge acceptance, not a fill.
"""

import os
from numbers import Integral
import re

import requests

from data.loader import (
    KISCredentials,
    KISDailyPriceLoader,
    KIS_REAL_BASE_URL,
    KIS_PAPER_BASE_URL,
)


class OrderSubmissionUncertain(RuntimeError):
    """A request may have reached KIS. Reconcile broker order history before retrying."""


class KISBroker:
    """Account lookup and explicit limit-order submission, with no automatic retries."""

    def __init__(
        self,
        credentials,
        account_no,
        product_code,
        *,
        mode="paper",
        enable_orders=False,
        session=None,
    ):
        if mode not in {"paper", "real"}:
            raise ValueError("mode must be paper or real.")
        expected_url = KIS_PAPER_BASE_URL if mode == "paper" else KIS_REAL_BASE_URL
        if credentials.base_url.rstrip("/") != expected_url:
            raise ValueError(
                "Credential base URL does not match the selected broker mode."
            )
        if not re.fullmatch(r"\d{8}", account_no) or not re.fullmatch(
            r"\d{2}", product_code
        ):
            raise ValueError(
                "Account must be 8 digits and product code must be 2 digits."
            )
        if type(enable_orders) is not bool:
            raise ValueError("enable_orders must be boolean.")
        self.credentials = credentials
        self.account_no = account_no
        self.product_code = product_code
        self.mode = mode
        self.enable_orders = enable_orders
        self.session = session or requests.Session()
        self.authentication = KISDailyPriceLoader(credentials)
        self.attempted_order_ids = set()

    @classmethod
    def from_config(cls, config, *, mode=None):
        settings = config["broker"]
        mode = mode or settings["mode"]
        api = config["kis_api"]
        credentials = KISCredentials.from_env(
            environment=mode,
            app_key_env=api["app_key_env"],
            app_secret_env=api["app_secret_env"],
            access_token_env=api["access_token_env"],
        )
        account = os.environ.get(settings["account_env"], "")
        product = os.environ.get(settings["product_code_env"], "")
        return cls(
            credentials,
            account,
            product,
            mode=mode,
            enable_orders=settings["allow_orders"],
        )

    def _headers(self, transaction_id, *, continuation=""):
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.authentication._get_access_token()}",
            "appkey": self.credentials.app_key,
            "appsecret": self.credentials.app_secret,
            "tr_id": transaction_id,
            "custtype": "P",
            "tr_cont": continuation,
        }

    def get_balance(self):
        """Read all available position pages and the last account totals page."""
        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        transaction = "VTTC8434R" if self.mode == "paper" else "TTTC8434R"
        fk = nk = continuation = ""
        positions = []
        seen = set()
        for _ in range(50):
            response = self.session.get(
                self.credentials.base_url + path,
                headers=self._headers(transaction, continuation=continuation),
                params={
                    "CANO": self.account_no,
                    "ACNT_PRDT_CD": self.product_code,
                    "AFHR_FLPR_YN": "N",
                    "OFL_YN": "",
                    "INQR_DVSN": "02",
                    "UNPR_DVSN": "01",
                    "FUND_STTL_ICLD_YN": "N",
                    "FNCG_AMT_AUTO_RDPT_YN": "N",
                    "PRCS_DVSN": "00",
                    "CTX_AREA_FK100": fk,
                    "CTX_AREA_NK100": nk,
                },
                timeout=self.credentials.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("rt_cd") != "0":
                raise RuntimeError(
                    f"KIS balance request rejected: {payload.get('msg_cd', 'unknown')}"
                )
            positions.extend(payload.get("output1") or [])
            if response.headers.get("tr_cont") not in {"M", "F"}:
                return {"positions": positions, "totals": payload.get("output2") or []}
            fk, nk = payload.get("ctx_area_fk100", ""), payload.get(
                "ctx_area_nk100", ""
            )
            if (fk, nk) in seen or not (fk or nk):
                raise RuntimeError("KIS balance pagination did not advance.")
            seen.add((fk, nk))
            continuation = "N"
        raise RuntimeError(
            "KIS balance page limit exceeded; result would be incomplete."
        )

    def submit_limit_order(self, symbol, side, quantity, price, *, client_order_id):
        """Submit an explicit integer-price limit order. No local fill is assumed.

        Duplicate IDs are blocked within this process. Persistent reconciliation,
        cancellations and a live trading scheduler are outside this adapter.
        """
        if not self.enable_orders:
            raise PermissionError(
                "Order submission is disabled: broker.allow_orders is false."
            )
        if not re.fullmatch(r"\d{6}", symbol) or side not in {"buy", "sell"}:
            raise ValueError("Use a six-digit stock symbol and side buy/sell.")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            for value in (quantity, price)
        ):
            raise ValueError(
                "Limit order quantity and price must be positive integers."
            )
        if not isinstance(client_order_id, str) or not client_order_id.strip():
            raise ValueError("A nonempty client_order_id is required.")
        if client_order_id in self.attempted_order_ids:
            raise ValueError(
                "This client_order_id was already attempted; reconcile before another order."
            )
        prefix = "V" if self.mode == "paper" else "T"
        transaction = prefix + ("TTC0012U" if side == "buy" else "TTC0011U")
        headers = self._headers(transaction)
        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.product_code,
            "PDNO": symbol,
            "ORD_DVSN": "00",
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
            "EXCG_ID_DVSN_CD": "KRX",
            "SLL_TYPE": "01" if side == "sell" else "",
            "CNDT_PRIC": "",
        }
        self.attempted_order_ids.add(client_order_id)
        try:
            response = self.session.post(
                self.credentials.base_url
                + "/uapi/domestic-stock/v1/trading/order-cash",
                headers=headers,
                json=body,
                timeout=self.credentials.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise OrderSubmissionUncertain(
                "Order status is unknown. Check KIS order history before retrying."
            ) from exc
        if payload.get("rt_cd") != "0":
            raise RuntimeError(
                f"KIS order rejected: {payload.get('msg_cd', 'unknown')}"
            )
        output = payload.get("output") or {}
        if not output.get("ODNO"):
            raise OrderSubmissionUncertain(
                "KIS returned no order number; reconcile before retrying."
            )
        return {
            "client_order_id": client_order_id,
            "order_number": output["ODNO"],
            "status": "accepted",
            "output": output,
        }
