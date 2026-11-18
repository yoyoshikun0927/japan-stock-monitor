"""Private-use, no-notification Japanese equity reference monitor.

Market data via unofficial Yahoo/yfinance interface. Generated results are AES-GCM
encrypted before they enter the GitHub Pages artifact. Never publish unencrypted
market data, credentials, holdings, or order information.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import jpholiday
import numpy as np
import pandas as pd
import yfinance as yf
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'docs'
JST = ZoneInfo('Asia/Tokyo')
KDF_ITERATIONS = 600_000


def is_trading_session(now: dt.datetime) -> bool:
    now = now.astimezone(JST)
    d = now.date()
    if d.weekday() >= 5 or jpholiday.is_holiday(d) or (d.month, d.day) in {(1, 2), (1, 3), (12, 31)}:
        return False
    m = now.hour * 60 + now.minute
    return 540 <= m <= 690 or 750 <= m <= 930


def number(x, digits: int = 2):
    return round(float(x), digits) if pd.notna(x) and np.isfinite(x) else None


def rci(close: pd.Series, window: int) -> pd.Series:
    def calc(a: np.ndarray) -> float:
        rank = pd.Series(a).rank(method='average').to_numpy()
        return (1 - 6 * ((rank - np.arange(1, window + 1)) ** 2).sum() / (window * (window * window - 1))) * 100
    return close.rolling(window).apply(calc, raw=True)


def metrics(df: pd.DataFrame, prefix: str) -> dict:
    close = pd.to_numeric(df['Close'], errors='coerce').dropna()
    if len(close) < 80:
        raise ValueError(f'確定{prefix}足80本未満')
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).where(loss != 0, 100.0)
    r9, r26 = rci(close, 9), rci(close, 26)
    return {
        prefix + 'close': number(close.iloc[-1]),
        prefix + 'ma25': number(close.rolling(25).mean().iloc[-1]),
        prefix + 'ma75': number(close.rolling(75).mean().iloc[-1]),
        prefix + 'rci9': number(r9.iloc[-1]),
        prefix + 'rci26': number(r26.iloc[-1]),
        prefix + 'rsi14': number(rsi.iloc[-1]),
        prefix + 'macd': number(macd.iloc[-1], 3),
        prefix + 'signal': number(signal.iloc[-1], 3),
        prefix + 'rci9_rising': bool(r9.iloc[-1] > r9.iloc[-2]),
        prefix + 'rci26_rising': bool(r26.iloc[-1] > r26.iloc[-2]),
    }


def get_data(code: str, now: dt.datetime) -> dict:
    ticker = yf.Ticker(f'{code}.T')
    intra = ticker.history(period='7d', interval='5m', auto_adjust=False)
    if intra.empty:
        raise ValueError('5分足データなし')
    intra = intra.sort_index()
    # Ignore incomplete latest candle; if upstream market data are delayed, this
    # may skip more recent bars conservatively rather than guessing they closed.
    intra = intra[intra.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now)]
    if len(intra) < 80:
        raise ValueError('確定5分足80本未満')
    daily = ticker.history(period='1y', interval='1d', auto_adjust=False)
    if daily.empty:
        raise ValueError('日足データなし')
    daily = daily.sort_index()
    now_jst = now.astimezone(JST)
    # Daily bar for today's session is not confirmed until after the close.
    if now_jst.hour < 16 and daily.index[-1].astimezone(JST).date() == now_jst.date():
        daily = daily.iloc[:-1]
    if len(daily) < 80:
        raise ValueError('確定日足80本未満')
    last_bar = intra.index[-1].to_pydatetime().astimezone(JST)
    age = max(0, int((now_jst - last_bar).total_seconds() // 60) - 5)
    data = {**metrics(intra, 'intra_'), **metrics(daily, 'daily_'),
            'bar_time': last_bar.isoformat(), 'observed_lag_minutes': age,
            'price': number(intra['Close'].iloc[-1])}
    # This is a technical observation score, not an investment rating.
    score = 0
    if data['daily_close'] and data['daily_ma25'] and data['daily_close'] > data['daily_ma25']:
        score += 10
    if data['daily_ma25'] and data['daily_ma75'] and data['daily_ma25'] > data['daily_ma75']:
        score += 10
    if data['intra_rci9_rising']:
        score += 10
    if data['intra_rci26_rising']:
        score += 5
    if data['intra_rci9'] is not None and -80 <= data['intra_rci9'] <= 0:
        score += 5
    if data['intra_macd'] is not None and data['intra_signal'] is not None and data['intra_macd'] > data['intra_signal']:
        score += 10
    data['watch_score_50'] = score
    if age > 30:
        data.update(status='stale', reason='観測遅延30分超。新規判断不可')
    else:
        data.update(status='reference_only', reason='非公式・遅延不定。売買判断に単独使用しない')
    return data


def encrypt_payload(payload: dict, passphrase: str) -> dict:
    if len(passphrase) < 16:
        raise ValueError('MONITOR_PASSPHRASE は16文字以上にしてください')
    salt, nonce = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=KDF_ITERATIONS).derive(passphrase.encode('utf-8'))
    message = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    ciphertext = AESGCM(key).encrypt(nonce, message, None)
    b64 = lambda b: base64.b64encode(b).decode('ascii')
    return {'version': 1, 'cipher': 'AES-256-GCM', 'kdf': 'PBKDF2-SHA256',
            'iterations': KDF_ITERATIONS, 'salt': b64(salt),
            'nonce': b64(nonce), 'ciphertext': b64(ciphertext)}


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    if '--smoke-test' in sys.argv:
        sample = encrypt_payload({'test': True, 'generated_at': now.isoformat()}, 'some-example-passphrase-at-least-16-chars')
        assert sample['cipher'] == 'AES-256-GCM'
        print('SMOKE_TEST_OK')
        return
    if '--force' not in sys.argv and not is_trading_session(now):
        print('取引時間外または休場日。取得しません')
        return
    passphrase = os.environ.get('MONITOR_PASSPHRASE', '')
    if len(passphrase) < 16:
        raise SystemExit('MONITOR_PASSPHRASE を GitHub Actions の Repository secrets に16文字以上で設定してください')
    symbols = json.loads((ROOT / 'symbols.json').read_text(encoding='utf-8'))
    results = []
    for idx, item in enumerate(symbols, 1):
        row = {'code': item['code'], 'name': item.get('name', item['code']), 'status': 'unavailable'}
        try:
            row.update(get_data(item['code'], dt.datetime.now(dt.timezone.utc)))
        except Exception as exc:
            # Do not output personal user data or secrets into the workflow log.
            row['reason'] = str(exc)[:120]
        results.append(row)
        if idx < len(symbols):
            time.sleep(0.3)
    payload = {'generated_at': now.astimezone(JST).isoformat(),
               'data_provider': 'Yahoo Finance/yfinance (非公式・遅延不定)',
               'notice': '参考用の技術観測。投資判断の推奨ではありません。通知・自動注文なし。',
               'results': results}
    OUT.mkdir(exist_ok=True)
    (OUT / 'latest.enc.json').write_text(
        json.dumps(encrypt_payload(payload, passphrase), ensure_ascii=False), encoding='utf-8')
    # Nothing unencrypted is written to public Pages content.
    print('分析完了：対象', len(results), 'データ取得', sum(r['status'] != 'unavailable' for r in results),
          '暗号化公開データを生成')


if __name__ == '__main__':
    main()
