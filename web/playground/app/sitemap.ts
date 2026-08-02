import type { MetadataRoute } from "next"

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date()
  return [
    {
      url: "https://surgeforecast.com",
      lastModified,
      changeFrequency: "hourly",
      priority: 1,
    },
    {
      url: "https://surgeforecast.com/status",
      lastModified,
      changeFrequency: "hourly",
      priority: 0.8,
    },
    {
      url: "https://surgeforecast.com/methodology",
      lastModified,
      changeFrequency: "monthly",
      priority: 0.7,
    },
  ]
}
