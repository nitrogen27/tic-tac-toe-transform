import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/**/*.test.mjs'],
    timeout: 120000, // 2 минуты для тестов обучения
    testTimeout: 120000,
  },
});

