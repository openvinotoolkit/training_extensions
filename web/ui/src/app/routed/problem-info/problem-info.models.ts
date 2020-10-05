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
  unit: string;
  key: string;
  value: any;
}

export interface IModel {
  id: string;
  metrics?: { [key: string]: IMetric[] };
  name: string;
  status: string;
  tensorBoardLogDir: string;
  dirPath: string;
  showOnChart: boolean;
}

export interface IProblem {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  type: string;
  imagesUrls?: string[];
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
  status: string;
  showOnChart: boolean;
}
