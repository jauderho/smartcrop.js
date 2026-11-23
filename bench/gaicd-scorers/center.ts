// Baseline: prefer large crops close to the image centre.
// score = -(centre distance / image diagonal) + 0.1 * (crop area / image area)
export async function scoreCandidates(
  _imagePath: string,
  width: number,
  height: number,
  candidates: [number, number, number, number][]
): Promise<number[]> {
  const diag = Math.hypot(width, height);
  return candidates.map(([x1, y1, x2, y2]) => {
    const d = Math.hypot((x1 + x2) / 2 - width / 2, (y1 + y2) / 2 - height / 2);
    return -d / diag + (0.1 * ((x2 - x1) * (y2 - y1))) / (width * height);
  });
}
