/**
 * Copyright (c) 2020 Intel Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
