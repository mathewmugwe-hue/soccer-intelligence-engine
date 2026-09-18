import { FixtureInput } from '../../src/types';

export function parseUniversalSlip(rawText: string): FixtureInput[] {
  if (!rawText || !rawText.trim()) return [];
  const text = rawText.trim();

  // 1. JSON Array
  if (text.startsWith("[") && text.endsWith("]")) {
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed.map((item, idx) => ({
          id: item.id || `match-${idx + 1}`,
          home: (item.home || item.homeTeam || "").trim(),
          away: (item.away || item.awayTeam || "").trim(),
          league: item.league || "Premier League",
          match_date: item.match_date || item.date || undefined,
          odds_home: parseFloat(item.odds_home || item.odds1 || item.oh) || null,
          odds_draw: parseFloat(item.odds_draw || item.oddsX || item.od) || null,
          odds_away: parseFloat(item.odds_away || item.odds2 || item.oa) || null,
          is_neutral: Boolean(item.is_neutral || item.neutral),
          tournament_phase: item.tournament_phase || item.phase || "league",
        })).filter(f => f.home && f.away);
      }
    } catch {
      // Fall through to text parsers
    }
  }

  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const fixtures: FixtureInput[] = [];

  // 2. ODIBETS
  if (/ID:\s*\d+/i.test(text)) {
    let i = 0;
    while (i < lines.length) {
      if (/ID:\s*\d+/i.test(lines[i])) {
        i++;
        if (i < lines.length) {
          const home = lines[i++];
          if (i < lines.length) {
            const away = lines[i++];
            let oh: number | null = null;
            let od: number | null = null;
            let oa: number | null = null;
            while (i < lines.length && !/ID:\s*\d+/i.test(lines[i])) {
              const cur = lines[i];
              if (/^Home$/i.test(cur) && i + 1 < lines.length && /^\d+(\.\d+)?$/.test(lines[i + 1])) {
                oh = parseFloat(lines[++i]);
              } else if (/^Draw$/i.test(cur) && i + 1 < lines.length && /^\d+(\.\d+)?$/.test(lines[i + 1])) {
                od = parseFloat(lines[++i]);
              } else if (/^Away$/i.test(cur) && i + 1 < lines.length && /^\d+(\.\d+)?$/.test(lines[i + 1])) {
                oa = parseFloat(lines[++i]);
              } else {
                i++;
              }
            }
            if (home && away && !home.includes("ID:") && !away.includes("ID:")) {
              fixtures.push({
                id: `odi-${fixtures.length + 1}`,
                home: home.trim(),
                away: away.trim(),
                league: "Standard League",
                odds_home: oh,
                odds_draw: od,
                odds_away: oa,
              });
            }
            continue;
          }
        }
      }
      i++;
    }
    if (fixtures.length > 0) return fixtures;
  }

  // 3. BETIKA
  if (text.includes("•") || /\+\d+\s*Markets/i.test(text)) {
    let curLeague = "Premier League";
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (line.includes("•") && !line.includes("STARTS IN")) {
        curLeague = line.replace(/International Clubs\s*•\s*/i, "").trim();
        i++;
        if (i < lines.length && /^\d{2}\/\d{2}/.test(lines[i])) i++;
        if (i < lines.length) {
          const home = lines[i++];
          if (i < lines.length) {
            const away = lines[i++];
            const oddsList: number[] = [];
            while (i < lines.length && !lines[i].includes("•") && !/\+\d+\s*Markets/i.test(lines[i])) {
              const val = parseFloat(lines[i]);
              if (!isNaN(val) && val > 1.0 && val < 100.0) {
                oddsList.push(val);
              }
              i++;
            }
            if (home && away) {
              fixtures.push({
                id: `betika-${fixtures.length + 1}`,
                home: home.trim(),
                away: away.trim(),
                league: curLeague,
                odds_home: oddsList[0] || null,
                odds_draw: oddsList[1] || null,
                odds_away: oddsList[2] || null,
              });
            }
            continue;
          }
        }
      }
      i++;
    }
    if (fixtures.length > 0) return fixtures;
  }

  // 4. SPORTPESA
  if (/#\d+/.test(text)) {
    let curLeague = "Premier League";
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (line.includes("/") && /\(\d+\)$/.test(line)) {
        curLeague = line.replace(/\(\d+\)$/, "").replace(/International Clubs\s*\/\s*/i, "").trim();
        i++;
        continue;
      }
      if (/^#\d+$/.test(line)) {
        let idx = i - 1;
        if (idx >= 0 && (lines[idx].includes("STARTS IN") || /^\d{2}\/\d{2}/.test(lines[idx]))) {
          idx--;
        }
        const away = idx >= 0 ? lines[idx--] : "";
        const home = idx >= 0 ? lines[idx--] : "";

        let oh: number | null = null;
        let od: number | null = null;
        let oa: number | null = null;
        let fwd = i + 1;
        while (fwd < lines.length && !/^#\d+$/.test(lines[fwd]) && !/\(\d+\)$/.test(lines[fwd])) {
          if (lines[fwd] === "1" && fwd + 1 < lines.length && /^\d+(\.\d+)?$/.test(lines[fwd + 1])) {
            oh = parseFloat(lines[fwd + 1]);
            fwd += 2;
          } else if ((lines[fwd] === "X" || lines[fwd] === "x") && fwd + 1 < lines.length && /^\d+(\.\d+)?$/.test(lines[fwd + 1])) {
            od = parseFloat(lines[fwd + 1]);
            fwd += 2;
          } else if (lines[fwd] === "2" && fwd + 1 < lines.length && /^\d+(\.\d+)?$/.test(lines[fwd + 1])) {
            oa = parseFloat(lines[fwd + 1]);
            fwd += 2;
          } else {
            fwd++;
          }
        }
        if (home && away && home !== "🔥 HOT" && away !== "🔥 HOT") {
          fixtures.push({
            id: `sp-${fixtures.length + 1}`,
            home: home.trim(),
            away: away.trim(),
            league: curLeague,
            odds_home: oh,
            odds_draw: od,
            odds_away: oa,
          });
        }
        i = fwd;
        continue;
      }
      i++;
    }
    if (fixtures.length > 0) return fixtures;
  }

  // 5. General Line-by-Line matching
  for (let idx = 0; idx < lines.length; idx++) {
    const l = lines[idx];
    if (/\bvs\b/i.test(l) || (l.includes("-") && !l.includes("-->"))) {
      const parts = l.split(/[,;\t|]/).map(p => p.trim());
      const matchPart = parts[0];
      let h = "";
      let a = "";
      if (/\bvs\b/i.test(matchPart)) {
        const split = matchPart.split(/\bvs\b/i);
        h = split[0];
        a = split[1];
      } else if (matchPart.includes("-")) {
        const split = matchPart.split("-");
        h = split[0];
        a = split[1];
      }

      if (h && a) {
        fixtures.push({
          id: `line-${fixtures.length + 1}`,
          home: h.trim(),
          away: a.trim(),
          league: parts[4] || "Standard League",
          odds_home: parseFloat(parts[1]) || null,
          odds_draw: parseFloat(parts[2]) || null,
          odds_away: parseFloat(parts[3]) || null,
        });
      }
    }
  }

  return fixtures;
}
