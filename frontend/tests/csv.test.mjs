import { test } from "node:test";
import assert from "node:assert";
import { jobsToCsv } from "../lib/csv.ts";

function job(overrides = {}) {
  return {
    id: "job1", mode: "closeup", status: "done", progress: 1, message: null,
    batch_id: null, filename: "sample.png", created_at: "2026-07-05T10:00:00",
    result: {
      mode: "closeup",
      verdict: {
        ore_class: "ordinary", text: "заключение",
        metrics: {
          sulfide_frac: 0.21, magnetite_frac: 0.05, matrix_frac: 0.74,
          talc_frac: 0.03, talc_share_est: 0.04, fine_share: 0.3,
          confidence: 0.71, undetermined_fraction: 0.08,
        },
      },
      sort: null,
    },
    ...overrides,
  };
}

test("jobsToCsv writes a header row and one data row per job", () => {
  const csv = jobsToCsv([job()]);
  const lines = csv.split("\r\n");
  assert.strictEqual(lines.length, 2);
  assert.strictEqual(
    lines[0],
    "job_id,filename,mode,status,ore_class,sulfide_frac,magnetite_frac,matrix_frac,talc_frac,talc_share_est,fine_share,confidence,undetermined_fraction,created_at",
  );
  assert.strictEqual(
    lines[1],
    "job1,sample.png,closeup,done,ordinary,0.21,0.05,0.74,0.03,0.04,0.3,0.71,0.08,2026-07-05T10:00:00",
  );
});

test("jobsToCsv leaves missing metrics blank instead of crashing", () => {
  const pano = job({
    id: "job2", mode: "panorama", filename: "pano.png",
    result: {
      mode: "panorama",
      verdict: { ore_class: "review", text: "", metrics: { talc_frac: 0.01, confidence: 0.4 } },
      sort: null,
    },
  });
  const row = jobsToCsv([pano]).split("\r\n")[1];
  assert.strictEqual(
    row,
    "job2,pano.png,panorama,done,review,,,,0.01,,,0.4,,2026-07-05T10:00:00",
  );
});

test("jobsToCsv escapes commas, quotes and newlines in filenames", () => {
  const weird = job({ id: "job3", filename: 'a, "tricky"\nname.png' });
  const row = jobsToCsv([weird]).split("\r\n")[1];
  assert.ok(row.startsWith('job3,"a, ""tricky""\nname.png",closeup,done,ordinary,'));
});

test("jobsToCsv handles a job with no result yet (queued/running)", () => {
  const pending = job({ id: "job4", status: "queued", filename: "later.png", result: null });
  const row = jobsToCsv([pending]).split("\r\n")[1];
  assert.strictEqual(row, "job4,later.png,closeup,queued,,,,,,,,,,2026-07-05T10:00:00");
});

test("jobsToCsv concatenates multiple jobs as separate rows", () => {
  const csv = jobsToCsv([job({ id: "a" }), job({ id: "b", filename: "second.png" })]);
  assert.strictEqual(csv.split("\r\n").length, 3);
});
