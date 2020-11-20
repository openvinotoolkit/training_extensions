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

export interface IBuild {
  id: string;
  problemId: string;
  name: string;
  status: string;
}

export interface IMetric {
  id: string;
  displayName: string;
  key: string;
  value: any;
  unit: string;
}

export interface IEvaluate {
  metrics: IMetric[];
  status: string;
}

export interface IModel {
  id: string;
  dir: string;
  evaluates?: { [key: string]: IEvaluate };
  name: string;
  showOnChart: boolean;
  status: string;
  tensorBoardLogDir: string;
}

export interface IProblem {
  id: string;
  description: string;
  imagesUrls?: string[];
  subtitle: string;
  title: string;
  type: string;
}

export interface IModelsMetricsTableData {
  [key: string]: any;
}

export interface IModelsMetricsTableColumn {
  key: string;
  label: string;
  unit?: string;
}

export interface ITableCell {
  value: number;
  opacity: number;
}

export interface ITableNameCell {
  value: string;
}

export interface IModelTableRow {
  [key: string]: ITableCell | ITableNameCell | string | boolean;

  id: string;
  name: ITableNameCell;
  showOnChart: boolean;
  trainStatus: string;
  evalStatus: string;
}
