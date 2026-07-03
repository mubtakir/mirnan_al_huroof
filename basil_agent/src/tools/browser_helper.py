import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SovereignBrowser")

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print(json.dumps({"status": "error", "message": "Playwright not installed. Run: pip install playwright && python -m playwright install chromium"}))
    sys.exit(1)

class SovereignBrowserAgent:
    def __init__(self, workspace_root: str, headless: bool = True):
        self.workspace_root = Path(workspace_root)
        self.sessions_dir = self.workspace_root / ".browser_sessions"
        self.screenshots_dir = self.workspace_root / ".agent_screenshots" / "browser"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._session_name = "default"
        self.headless = headless

    def _launch(self):
        if self._browser:
            return
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
        )

        session_file = self.sessions_dir / f"{self._session_name}.json"
        storage_state = str(session_file) if session_file.exists() else None

        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            storage_state=storage_state,
            accept_downloads=True,
        )
        self._page = self._context.new_page()

    def _save_session(self):
        if self._context:
            session_file = self.sessions_dir / f"{self._session_name}.json"
            self._context.storage_state(path=str(session_file))

    def close(self):
        self._save_session()
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()
        self._browser = None
        self._context = None
        self._page = None
        self._pw = None

    def navigate(self, url: str) -> dict:
        try:
            self._launch()
            if not url.startswith("http"):
                url = "https://" + url
            response = self._page.goto(url, timeout=30000, wait_until="domcontentloaded")
            status = response.status if response else "unknown"
            title = self._page.title()
            return {"status": "success", "url": self._page.url, "http_status": status, "title": title}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def click(self, selector: str) -> dict:
        try:
            self._launch()
            self._page.click(selector, timeout=8000)
            time.sleep(0.5)
            return {"status": "success", "message": f"Clicked on {selector}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def fill(self, selector: str, value: str) -> dict:
        try:
            self._launch()
            self._page.fill(selector, value, timeout=8000)
            return {"status": "success", "message": f"Filled {selector} with value"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_text(self, max_length: int = 6000) -> dict:
        try:
            self._launch()
            text = self._page.evaluate("""() => {
                const elements = document.querySelectorAll('script, style, noscript, nav, footer, header, aside, .ad, .ads, [role="banner"]');
                elements.forEach(el => el.remove());
                return document.body ? document.body.innerText : '';
            }""")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            clean = "\n".join(lines)
            if len(clean) > max_length:
                clean = clean[:max_length] + "\n... [truncated]"
            return {"status": "success", "url": self._page.url, "title": self._page.title(), "content": clean}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_links(self) -> dict:
        try:
            self._launch()
            links = self._page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({text: a.innerText.trim(), href: a.href}))
                    .filter(l => l.href.startsWith('http') && l.text.length > 0)
                    .slice(0, 30);
            }""")
            return {"status": "success", "links": links}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def screenshot(self, label: str = "state") -> dict:
        try:
            self._launch()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.screenshots_dir / f"browser_{label}_{ts}.png"
            self._page.screenshot(path=str(path), full_page=False)
            return {"status": "success", "path": str(path.resolve())}
        except Exception as e:
            return {"status": "error", "message": str(e)}

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "message": "Usage: python browser_helper.py <workspace_root> <command> [args...]"}))
        sys.exit(1)

    workspace_root = sys.argv[1]
    command = sys.argv[2]
    
    agent = SovereignBrowserAgent(workspace_root, headless=True)
    result = {"status": "error", "message": f"Unknown command: {command}"}

    try:
        if command == "navigate":
            if len(sys.argv) < 4:
                result = {"status": "error", "message": "Missing URL argument"}
            else:
                result = agent.navigate(sys.argv[3])
        elif command == "click":
            if len(sys.argv) < 4:
                result = {"status": "error", "message": "Missing selector argument"}
            else:
                result = agent.click(sys.argv[3])
        elif command == "fill":
            if len(sys.argv) < 5:
                result = {"status": "error", "message": "Missing selector and/or value argument"}
            else:
                result = agent.fill(sys.argv[3], sys.argv[4])
        elif command == "text":
            result = agent.get_text()
        elif command == "links":
            result = agent.get_links()
        elif command == "screenshot":
            label = sys.argv[3] if len(sys.argv) >= 4 else "state"
            result = agent.screenshot(label)
    except Exception as e:
        result = {"status": "error", "message": f"Execution error: {str(e)}"}
    finally:
        agent.close()

    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
