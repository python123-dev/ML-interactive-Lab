import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import Database from "better-sqlite3";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialize SQLite Database
const db = new Database("ml_lab.db");
db.exec(`
  CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    algo_id TEXT,
    algo_name TEXT,
    dataset_id TEXT,
    dataset_name TEXT,
    features TEXT,
    target TEXT,
    hyperparams TEXT,
    metrics TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API Routes
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", message: "ML Interactive Lab API is active" });
  });

  // Experiment History APIs
  app.get("/api/history", (req, res) => {
    try {
      const rows = db.prepare("SELECT * FROM experiments ORDER BY created_at DESC").all();
      res.json(rows.map((row: any) => ({
        ...row,
        features: JSON.parse(row.features),
        hyperparams: JSON.parse(row.hyperparams),
        metrics: JSON.parse(row.metrics)
      })));
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch history" });
    }
  });

  app.post("/api/history", (req, res) => {
    const { algo_id, algo_name, dataset_id, dataset_name, features, target, hyperparams, metrics } = req.body;
    try {
      const stmt = db.prepare(`
        INSERT INTO experiments (algo_id, algo_name, dataset_id, dataset_name, features, target, hyperparams, metrics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `);
      const result = stmt.run(
        algo_id,
        algo_name,
        dataset_id,
        dataset_name,
        JSON.stringify(features),
        target,
        JSON.stringify(hyperparams),
        JSON.stringify(metrics)
      );
      res.json({ id: result.lastInsertRowid });
    } catch (error) {
      res.status(500).json({ error: "Failed to save experiment" });
    }
  });

  app.delete("/api/history/:id", (req, res) => {
    try {
      db.prepare("DELETE FROM experiments WHERE id = ?").run(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to delete experiment" });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // Serve static files in production
    app.use(express.static(path.resolve(__dirname, "dist")));
    app.get("*", (req, res) => {
      res.sendFile(path.resolve(__dirname, "dist", "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
