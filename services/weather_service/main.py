
from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI(
    title="LuaAI - Weather Service",
    description="Dich vu thoi tiet va canh bao cho nguoi trong lua.",
    version="1.0.0",
)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

SAPA_LOCATION = {
    "name": "Sa Pa, Lao Cai",
    "latitude": 22.3364,
    "longitude": 103.8438,
}


async def fetch_weather():
    params = {
        "latitude": SAPA_LOCATION["latitude"],
        "longitude": SAPA_LOCATION["longitude"],
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),
        "daily": "temperature_2m_min,temperature_2m_max",
        "forecast_days": 3,
        "timezone": "Asia/Bangkok",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                OPEN_METEO_URL,
                params=params,
            )
            response.raise_for_status()
            return response.json()

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Weather provider is temporarily unavailable.",
        ) from exc


@app.get("/health")
async def health_check():
    return {
        "service": "weather_service",
        "status": "healthy",
    }


@app.get("/weather/sapa")
async def get_sapa_weather():
    data = await fetch_weather()

    return {
        "location": SAPA_LOCATION["name"],
        "source": "Open-Meteo",
        "current_weather": data.get("current"),
        "daily_forecast": data.get("daily"),
    }


@app.get("/weather/alerts")
async def get_weather_alerts():
    data = await fetch_weather()
    daily = data.get("daily") or {}

    dates = daily.get("time") or []
    min_temps = daily.get("temperature_2m_min") or []

    alerts = []

    for date, temp in zip(dates, min_temps):
        if temp is None:
            continue

        if temp <= 10:
            alerts.append({
                "date": date,
                "level": "high",
                "temperature_min": temp,
                "message": "Canh bao lanh: can kiem tra dieu kien canh tac.",
            })

        elif temp <= 15:
            alerts.append({
                "date": date,
                "level": "caution",
                "temperature_min": temp,
                "message": "Nhiet do thap: can theo doi tinh trang cay lua.",
            })

    if not alerts:
        alerts.append({
            "level": "info",
            "message": "Chua phat hien nguong canh bao lanh thu nghiem.",
        })

    return {
        "location": SAPA_LOCATION["name"],
        "source": "Open-Meteo",
        "warning_rule": "Experimental temperature thresholds",
        "alerts": alerts,
    }
