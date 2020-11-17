/**
 * @overview
 * @copyright (c) JSC Intel A/O
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
