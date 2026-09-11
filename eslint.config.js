const expoConfig = require('eslint-config-expo/flat');
const prettier = require('eslint-config-prettier');

module.exports = [
  ...expoConfig,
  prettier,
  {
    ignores: ['node_modules/**', 'dist/**', '.expo/**', 'coverage/**', 'android/**', 'ios/**'],
  },
  {
    rules: {
      'no-console': ['warn', { allow: ['warn', 'error', 'info'] }],
    },
  },
  {
    // Jest config/setup files run in Node with the Jest globals available.
    files: ['jest.setup.js', 'jest.config.js', '**/*.test.ts', '**/*.test.tsx'],
    languageOptions: {
      globals: {
        jest: 'readonly',
        describe: 'readonly',
        it: 'readonly',
        test: 'readonly',
        expect: 'readonly',
        beforeAll: 'readonly',
        beforeEach: 'readonly',
        afterAll: 'readonly',
        afterEach: 'readonly',
        require: 'readonly',
        module: 'writable',
        console: 'readonly',
      },
    },
  },
];
