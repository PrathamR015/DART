/** Two-digit numbering used for questions and ranks: 1 -> "01". */
export function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n)
}

/** A probability (0..1) as a percentage with one decimal: 0.9714 -> "97.1%". */
export function percent(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`
}

export function plural(count: number, one: string, many: string = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`
}
