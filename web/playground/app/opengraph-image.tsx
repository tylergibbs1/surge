import { ImageResponse } from "next/og"

export const runtime = "edge"
export const alt = "Surge — open forecasts for the US power grid"
export const size = { width: 1200, height: 630 }
export const contentType = "image/png"

export default async function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          background:
            "radial-gradient(circle at 20% 0%, #1a2a4d 0%, #0a0a0a 50%)",
          color: "white",
          padding: "72px 80px",
          fontFamily: "sans-serif",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              width: 20,
              height: 20,
              borderRadius: 999,
              background: "#4F8DF7",
              boxShadow: "0 0 24px #4F8DF7",
            }}
          />
          <span style={{ fontSize: 28, letterSpacing: -0.5, opacity: 0.7 }}>
            surge
          </span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <h1
            style={{
              fontSize: 76,
              fontWeight: 600,
              letterSpacing: -2,
              lineHeight: 1.05,
              margin: 0,
              maxWidth: 980,
            }}
          >
            Open day-ahead load forecasts for the US power grid.
          </h1>
          <p
            style={{
              fontSize: 28,
              lineHeight: 1.4,
              opacity: 0.82,
              margin: 0,
              maxWidth: 900,
            }}
          >
            Chronos-2 fine-tuned on 7 years of public data. Test MASE 0.45
            across 7 ISOs — matches utility-internal accuracy. Free &
            open-source.
          </p>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-end",
            fontSize: 22,
            opacity: 0.6,
          }}
        >
          <span>github.com/tylergibbs1/surge</span>
          <span style={{ display: "flex", gap: 20 }}>
            <span>PJM · CAISO · ERCOT · MISO</span>
            <span>NYISO · ISO-NE · SPP</span>
          </span>
        </div>
      </div>
    ),
    { ...size },
  )
}
