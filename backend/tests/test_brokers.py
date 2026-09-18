"""Mocked HTTP tests only: never authenticate or submit a real KIS order."""

import unittest
from unittest.mock import Mock

import requests

from broker.paper_broker import PaperBroker
from broker.real_broker import KISBroker, OrderSubmissionUncertain
from data.loader import KISCredentials, KIS_PAPER_BASE_URL, KIS_REAL_BASE_URL


def broker(*, allow=False, mode="paper"):
    base_url = KIS_PAPER_BASE_URL if mode == "paper" else KIS_REAL_BASE_URL
    credentials = KISCredentials(
        "test-key", "test-secret", base_url=base_url, access_token="test-token"
    )
    return KISBroker(
        credentials, "12345678", "01", mode=mode, enable_orders=allow, session=Mock()
    )


def response(payload, continuation=""):
    output = Mock()
    output.json.return_value = payload
    output.headers = {"tr_cont": continuation}
    return output


class TestBrokers(unittest.TestCase):
    def test_local_paper_account_has_no_network_dependency(self):
        account = PaperBroker(1000, commission=0.01, tax=0.02, min_holding_days=0)
        fill = account.rebalance(1, 100)
        self.assertEqual(fill.shares_traded, 9)
        self.assertEqual(account.snapshot(100)["asset_value"], 991)
        account.finish_day()
        self.assertEqual(account.holding_days, 1)
        account.rebalance(0, 100)
        self.assertEqual(account.cash, 964)

    def test_disabled_orders_never_make_http_request(self):
        client = broker()
        with self.assertRaises(PermissionError):
            client.submit_limit_order("005930", "buy", 1, 70000, client_order_id="one")
        client.session.post.assert_not_called()

    def test_paper_endpoint_and_payload_and_duplicate_block(self):
        client = broker(allow=True)
        client.session.post.return_value = response(
            {"rt_cd": "0", "output": {"ODNO": "123"}}
        )
        result = client.submit_limit_order(
            "005930", "buy", 1, 70000, client_order_id="one"
        )
        self.assertEqual(result["status"], "accepted")
        kwargs = client.session.post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["tr_id"], "VTTC0012U")
        self.assertEqual(kwargs["json"]["ORD_DVSN"], "00")
        self.assertEqual(kwargs["json"]["ORD_QTY"], "1")
        self.assertTrue(
            client.session.post.call_args.args[0].startswith(KIS_PAPER_BASE_URL)
        )
        with self.assertRaises(ValueError):
            client.submit_limit_order("005930", "buy", 1, 70000, client_order_id="one")
        self.assertEqual(client.session.post.call_count, 1)

    def test_real_mode_cannot_use_paper_credentials(self):
        credentials = KISCredentials("test", "test", base_url=KIS_PAPER_BASE_URL)
        with self.assertRaises(ValueError):
            KISBroker(credentials, "12345678", "01", mode="real")

    def test_timeout_is_not_retried_and_blocks_duplicate_attempt(self):
        client = broker(allow=True)
        client.session.post.side_effect = requests.Timeout()
        with self.assertRaises(OrderSubmissionUncertain):
            client.submit_limit_order("005930", "buy", 1, 70000, client_order_id="one")
        with self.assertRaises(ValueError):
            client.submit_limit_order("005930", "buy", 1, 70000, client_order_id="one")
        self.assertEqual(client.session.post.call_count, 1)

    def test_balance_pagination_returns_every_page(self):
        client = broker()
        client.session.get.side_effect = [
            response(
                {
                    "rt_cd": "0",
                    "output1": [{"pdno": "005930"}],
                    "output2": [{}],
                    "ctx_area_fk100": "a",
                    "ctx_area_nk100": "b",
                },
                "M",
            ),
            response(
                {
                    "rt_cd": "0",
                    "output1": [{"pdno": "000660"}],
                    "output2": [{"cash": "100"}],
                }
            ),
        ]
        result = client.get_balance()
        self.assertEqual(len(result["positions"]), 2)
        self.assertEqual(result["totals"], [{"cash": "100"}])
        self.assertEqual(client.session.get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
