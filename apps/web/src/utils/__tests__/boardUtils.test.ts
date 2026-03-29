import { describe, test, expect } from "vitest";
import {
  flatToRowCol,
  rowColToFlat,
  colLabel,
  rowLabel,
  moveToString,
  emptyCells,
  checkWinner,
} from "../boardUtils";

describe("flatToRowCol", () => {
  test("center of 15x15", () => {
    expect(flatToRowCol(112, 15)).toEqual([7, 7]);
  });

  test("top-left corner", () => {
    expect(flatToRowCol(0, 15)).toEqual([0, 0]);
  });

  test("bottom-right corner", () => {
    expect(flatToRowCol(224, 15)).toEqual([14, 14]);
  });
});

describe("rowColToFlat", () => {
  test("round-trips with flatToRowCol", () => {
    for (const idx of [0, 7, 112, 224]) {
      const [r, c] = flatToRowCol(idx, 15);
      expect(rowColToFlat(r, c, 15)).toBe(idx);
    }
  });
});

describe("colLabel", () => {
  test("col 0 is A", () => expect(colLabel(0)).toBe("A"));
  test("col 7 is H", () => expect(colLabel(7)).toBe("H"));
  test("col 8 is J (skips I)", () => expect(colLabel(8)).toBe("J"));
  test("col 14 is P", () => expect(colLabel(14)).toBe("P"));
});

describe("rowLabel", () => {
  test("row 0 on 15x15 is 15", () => expect(rowLabel(0, 15)).toBe("15"));
  test("row 14 on 15x15 is 1", () => expect(rowLabel(14, 15)).toBe("1"));
});

describe("moveToString", () => {
  test("center of 15x15 is H8", () => expect(moveToString(112, 15)).toBe("H8"));
  test("top-left of 15x15 is A15", () => expect(moveToString(0, 15)).toBe("A15"));
});

describe("emptyCells", () => {
  test("correct length", () => expect(emptyCells(9).length).toBe(81));
  test("all zeros", () => expect(emptyCells(7).every((c) => c === 0)).toBe(true));
});

describe("checkWinner", () => {
  function makeCells(size: number): (0 | 1 | -1)[] {
    return new Array(size * size).fill(0);
  }

  test("horizontal 5 in a row", () => {
    const cells = makeCells(15);
    // Place 5 black stones in row 7, cols 3-7
    for (let c = 3; c <= 7; c++) cells[7 * 15 + c] = 1;
    expect(checkWinner(cells, 15, 7 * 15 + 5)).toBe(1); // check from middle
  });

  test("vertical 5 in a row", () => {
    const cells = makeCells(15);
    for (let r = 2; r <= 6; r++) cells[r * 15 + 4] = -1;
    expect(checkWinner(cells, 15, 4 * 15 + 4)).toBe(-1);
  });

  test("diagonal 5 in a row", () => {
    const cells = makeCells(15);
    for (let i = 0; i < 5; i++) cells[(3 + i) * 15 + (3 + i)] = 1;
    expect(checkWinner(cells, 15, 5 * 15 + 5)).toBe(1);
  });

  test("anti-diagonal 5 in a row", () => {
    const cells = makeCells(15);
    for (let i = 0; i < 5; i++) cells[(3 + i) * 15 + (10 - i)] = -1;
    expect(checkWinner(cells, 15, 5 * 15 + 8)).toBe(-1);
  });

  test("4 in a row is not enough", () => {
    const cells = makeCells(15);
    for (let c = 0; c < 4; c++) cells[7 * 15 + c] = 1;
    expect(checkWinner(cells, 15, 7 * 15 + 3)).toBeNull();
  });

  test("no winner on empty board", () => {
    expect(checkWinner(makeCells(15), 15, -1)).toBeNull();
  });

  test("win at board edge", () => {
    const cells = makeCells(15);
    for (let c = 0; c < 5; c++) cells[0 * 15 + c] = 1; // top row, first 5 cols
    expect(checkWinner(cells, 15, 0 * 15 + 2)).toBe(1);
  });

  test("win at bottom-right edge", () => {
    const cells = makeCells(15);
    for (let c = 10; c < 15; c++) cells[14 * 15 + c] = -1;
    expect(checkWinner(cells, 15, 14 * 15 + 12)).toBe(-1);
  });

  test("small board 7x7", () => {
    const cells = makeCells(7);
    for (let r = 1; r <= 5; r++) cells[r * 7 + 3] = 1;
    expect(checkWinner(cells, 7, 3 * 7 + 3)).toBe(1);
  });
});
