/**
 * @overview Production environment
 * @copyright (c) JSC Intel A/O
 */

const host = process?.env?.IDLP_HOST?.trim().length ? process.env.IDLP_HOST : 'localhost';

export const environment = {
  production: true,
  title: 'OpenVINO Training Extensions',
  tensorBoardUrl: `http://${host}:8002`,
  filebrowserUrl: `http://${host}:8003`,
  cvatUrl: `http://${host}:8080`,
  ws: `ws://${host}:8001/api/ws`,
  themes: {
    light: 'light-theme',
    dark: 'dark-theme',
  },
  themeDefault: 'dark-theme',
};
