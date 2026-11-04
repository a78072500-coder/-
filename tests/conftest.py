"""GitHub Actions 에서 실패한 테스트를 주석으로 남긴다.

로그는 로그인해야 볼 수 있지만 주석은 공개 API 로 읽을 수 있어서 원인을 바로 확인할 수 있다.
"""

import os


def _escape(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def pytest_terminal_summary(terminalreporter):
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    for report in terminalreporter.stats.get("failed", []):
        detail = report.longreprtext[-1500:]
        for _title, content in report.sections:
            detail += "\n" + content[-500:]
        terminalreporter.write_line(f"::error title={_escape(report.nodeid)}::{_escape(detail)}")
