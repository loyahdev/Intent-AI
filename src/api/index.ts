import express from "express";
import fetch from "node-fetch";
import bodyParser from "body-parser";
import { Configuration, OpenAIApi } from "openai";

const PORT = process.env.PORT || 4000;
const LOCAL_URL = process.env.LOCAL_INTENT_URL || "http://localhost:8001/predict";

const openai = new OpenAIApi(
  new Configuration({ apiKey: process.env.OPENAI_API_KEY })
);

const app = express();
app.use(bodyParser.json({ limit: "1mb" }));

app.post("/analyze", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "text required" });

    // 1. Local model
    const local = await fetch(LOCAL_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    }).then(r => r.json());

    // 2. GPT‑4o deep explanation
    const gpt = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.2,
      messages: [
        { role: "system", content: "You are an intent‑detection assistant." },
        {
          role: "user",
          content:
            `Text:\n"""${text}"""\n\n` +
            `Local flags: ${JSON.stringify(local)}\n\n` +
            "Explain the intent, give 0‑1 scores for each label, and list highlighted spans.\n" +
            'Return JSON { "scores": {...}, "explanation": "...", "highlights": [...] }.'
        }
      ]
    });

    const parsed = JSON.parse(gpt.choices[0].message.content as string);
    res.json({ ...parsed, local });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "analysis failed" });
  }
});

app.listen(PORT, () => console.log(`API running on :${PORT}`));