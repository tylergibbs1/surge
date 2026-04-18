const API = process.env.SURGE_API_URL ?? "http://127.0.0.1:8000"

export async function GET() {
  try {
    const r = await fetch(`${API}/health`, { cache: "no-store" })
    return new Response(await r.text(), {
      status: r.status,
      headers: { "content-type": "application/json" },
    })
  } catch (e) {
    return Response.json(
      { status: "error", message: "upstream unreachable", upstream: API },
      { status: 502 }
    )
  }
}
