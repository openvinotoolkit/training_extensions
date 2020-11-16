/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {Injectable} from '@angular/core';
import {environment} from '@environments/environment';
import {SetTheme} from '@idlp/root/idlp-root.actions';
import {IdlpRootState} from '@idlp/root/idlp-root.state';
import {
  IBuild,
  IMetric,
  IModel,
  IModelsMetricsTableColumn,
  IModelsMetricsTableData,
  IModelTableRow,
  IProblem,
  ITableCell
} from '@idlp/routed/problem-info/problem-info.models';
import {Utils} from '@idlp/utils/utils';
import {Action, Selector, State, StateContext, Store} from '@ngxs/store';
import {ImmutableSelector} from '@ngxs-labs/immer-adapter';
import {
  ModelDelete,
  Reset,
  UpdateActiveBuild,
  UpdateBuilds,
  UpdateModels,
  UpdateModelsMetricsTableColumns,
  UpdateModelsMetricsTableData,
  UpdateProblemDetails,
  UpdateProblemId,
  UpdateScatterData,
  UpdateScatterLayout,
  UpdateXAxis,
  UpdateXAxisActive,
  UpdateYAxis,
  UpdateYAxisActive
} from './problem-info.actions';

export class ProblemInfoStateModel {
  public problemId: string;
  public problem: IProblem;
  public models: IModel[];
  public builds: IBuild[];
  public modelsMetricsTableData: IModelTableRow[];
  public modelsMetricsTableColumns: IModelsMetricsTableColumn[];
  public xAxis: IModelsMetricsTableColumn[];
  public yAxis: IModelsMetricsTableColumn[];
  public xAxisActive: IModelsMetricsTableColumn;
  public yAxisActive: IModelsMetricsTableColumn;
  public activeBuild: IBuild;
  public scatterData: any[];
  public scatterLayout: any;
}

const defaults = {
  problemId: '',
  problem: null,
  models: [],
  builds: [],
  modelsMetricsTableData: [],
  modelsMetricsTableColumns: [],
  xAxis: [],
  yAxis: [],
  xAxisActive: null,
  yAxisActive: null,
  activeBuild: null,
  scatterData: [],
  scatterLayout: {},
};

// @dynamic
@State<ProblemInfoStateModel>({
  name: 'problemInfo',
  defaults
})
@Injectable()
export class ProblemInfoState {

  constructor(private store: Store) {
  }

  @Selector()
  @ImmutableSelector()
  static problem(state: ProblemInfoStateModel): IProblem {
    return state.problem;
  }

  @Selector()
  static models(state: ProblemInfoStateModel): IModel[] {
    return state.models;
  }

  @Selector()
  static builds(state: ProblemInfoStateModel): IBuild[] {
    return state.builds;
  }

  @Selector()
  static activeBuild(state: ProblemInfoStateModel): IBuild {
    return state.activeBuild;
  }

  @Selector()
  static problemId(state: ProblemInfoStateModel): string {
    return state.problemId;
  }

  @Selector()
  static modelsMetricsTableData(state: ProblemInfoStateModel): IModelsMetricsTableData[] {
    return state.modelsMetricsTableData;
  }

  @Selector()
  static modelsMetricsTableColumns(state: ProblemInfoStateModel): IModelsMetricsTableColumn[] {
    return state.modelsMetricsTableColumns;
  }

  @Selector()
  static xAxis(state: ProblemInfoStateModel): IModelsMetricsTableColumn[] {
    return state.xAxis;
  }

  @Selector()
  static yAxis(state: ProblemInfoStateModel): IModelsMetricsTableColumn[] {
    return state.yAxis;
  }

  @Selector()
  static xAxisActive(state: ProblemInfoStateModel): IModelsMetricsTableColumn {
    return state.xAxisActive;
  }

  @Selector()
  static yAxisActive(state: ProblemInfoStateModel): IModelsMetricsTableColumn {
    return state.yAxisActive;
  }

  @Selector()
  static scatterData(state: ProblemInfoStateModel): any {
    return state.scatterData;
  }

  @Selector()
  static scatterLayout(state: ProblemInfoStateModel): any {
    return state.scatterLayout;
  }

  @Action(Reset)
  reset({setState}: StateContext<ProblemInfoStateModel>): void {
    setState(defaults);
  }

  @Action(UpdateProblemDetails)
  updateProblemDetails({getState, patchState}: StateContext<ProblemInfoStateModel>, {data}: UpdateProblemDetails): void {
    patchState({
      problem: data
    });
  }

  @Action(UpdateModels)
  updateModels({getState, patchState, dispatch}: StateContext<ProblemInfoStateModel>, {data}: UpdateModels): void {
    const state = getState();
    if (JSON.stringify(state.models) === JSON.stringify(data.items)) {
      return;
    }
    patchState({
      models: data.items,
    });
    const buildId = state.activeBuild?.id;
    const columns = this.getColumnsFromModels(data.items);
    if (JSON.stringify(state.modelsMetricsTableColumns) !== JSON.stringify(columns)) {
      dispatch(new UpdateModelsMetricsTableColumns(columns));
    }
    if (buildId) {
      dispatch(new UpdateModelsMetricsTableData(data.items, columns, buildId));
    }
  }

  @Action(ModelDelete)
  modelDelete({getState, dispatch}: StateContext<ProblemInfoStateModel>, {data}: ModelDelete): void {
    const state = getState();
    const items = state.models.filter(model => model.id !== data.id);
    const total = items.length;
    dispatch(new UpdateModels({total, items}));
  }

  @Action(UpdateBuilds)
  updateBuilds({getState, patchState, dispatch}: StateContext<ProblemInfoStateModel>, {data}: UpdateBuilds): void {
    let builds = data.items.filter(build => build.name !== '__tmp__');
    const state = getState();
    if (builds.length < 1 || JSON.stringify(builds) === JSON.stringify(state.builds)) {
      return;
    }
    builds = builds.sort((buildA: IBuild, buildB: IBuild) => buildA.name?.trim().toLocaleLowerCase() > buildB.name?.trim().toLocaleLowerCase() ? -1 : 1);
    patchState({
      builds,
    });
    dispatch(new UpdateActiveBuild(builds[0]));
  }

  @Action(UpdateActiveBuild)
  updateActiveBuild({getState, patchState, dispatch}: StateContext<ProblemInfoStateModel>, {data}: UpdateActiveBuild): void {
    const state = getState();
    patchState({
      activeBuild: data,
    });
    if (state.models.length > 0 && state.modelsMetricsTableColumns.length > 0 && data.id) {
      dispatch(new UpdateModelsMetricsTableData(state.models, state.modelsMetricsTableColumns, data.id));
    }
  }

  @Action(UpdateProblemId)
  updateProblemId({patchState}: StateContext<ProblemInfoStateModel>, {data}: UpdateProblemId): void {
    patchState({
      problemId: data,
    });
  }

  @Action(UpdateModelsMetricsTableData)
  updateModelsMetricsTableData({getState, patchState, dispatch}: StateContext<ProblemInfoStateModel>, {models, columns, buildId}: UpdateModelsMetricsTableData): void {
    const getStatus = (model: IModel): string => {
      if (!['trainFinished', 'trainDefault'].includes(model.status)) {
        return model.status;
      }
      return model.evaluates[buildId]?.status || 'notEvaluated';
    };
    const rows: IModelTableRow[] = [];
    const metricsValues: { [key: string]: number[] } = {};
    for (const model of models) {
      const row: IModelTableRow = {
        id: model.id,
        name: {
          value: model.name,
        },
        status: getStatus(model),
        showOnChart: model.showOnChart,
      };
      for (const {key} of columns) {
        if (key === 'name') {
          continue;
        }
        const metricValue = model?.evaluates?.[buildId]?.metrics?.find((metric: IMetric) => metric.key === key)?.value;
        row[key] = {
          value: metricValue,
        };
        if (!metricsValues[key]) {
          metricsValues[key] = [];
        }
        if (metricValue !== undefined) {
          metricsValues[key].push(metricValue);
        }
      }
      rows.push(row);
    }

    for (const {key} of columns) {
      if (key === 'name') {
        continue;
      }
      const metricMaxValue = Math.max(...metricsValues[key]);
      const metricMinValue = Math.min(...metricsValues[key]);
      rows.forEach((row, i, arr) => {
        const metricValue = (row[key] as ITableCell).value;
        (arr[i][key] as ITableCell).opacity = Math.round((metricValue - metricMinValue) / (metricMaxValue - metricMinValue) * 10) / 10;
      });
    }
    const state = getState();
    if (JSON.stringify(state.modelsMetricsTableData) !== JSON.stringify(rows)) {
      patchState({
        modelsMetricsTableData: rows,
      });
      dispatch(new UpdateScatterData());
    }
  }

  @Action(UpdateModelsMetricsTableColumns)
  updateModelsMetricsTableColumns({getState, dispatch, patchState}: StateContext<ProblemInfoStateModel>, {columns}: UpdateModelsMetricsTableColumns): void {
    patchState({
      modelsMetricsTableColumns: columns,
    });
    dispatch(new UpdateXAxis());
    dispatch(new UpdateYAxis());
  }

  @Action(UpdateXAxis)
  updateXAxis({getState, patchState, dispatch}: StateContext<ProblemInfoStateModel>): void {
    const state = getState();
    let cols: IModelsMetricsTableColumn[] = [];
    state.modelsMetricsTableColumns.forEach((col) => {
      if (col.key === 'complexity' || col.key === 'size') {
        cols.push(col);
      }
    });
    cols = Utils.objectArraySort(cols, 'label');
    patchState({
      xAxis: cols,
    });
    dispatch(new UpdateXAxisActive(cols[0]));
  }

  @Action(UpdateYAxis)
  updateYAxis({getState, patchState, dispatch}: StateContext<ProblemInfoStateModel>): void {
    const state = getState();
    let cols: IModelsMetricsTableColumn[] = [];
    state.modelsMetricsTableColumns.forEach((col) => {
      if (col.key !== 'complexity' && col.key !== 'size' && col.key !== 'name') {
        cols.push(col);
      }
    });
    cols = Utils.objectArraySort(cols, 'label');
    patchState({
      yAxis: cols,
    });
    dispatch(new UpdateYAxisActive(cols[0]));
  }

  @Action(UpdateXAxisActive)
  updateXAxisActive({patchState, dispatch}: StateContext<ProblemInfoStateModel>, {xAxis}: UpdateXAxisActive): void {
    patchState({
      xAxisActive: xAxis,
    });
    dispatch(new UpdateScatterData());
  }

  @Action(UpdateYAxisActive)
  updateYAxisActive({patchState, dispatch}: StateContext<ProblemInfoStateModel>, {yAxis}: UpdateYAxisActive): void {
    patchState({
      yAxisActive: yAxis,
    });
    dispatch(new UpdateScatterData());
  }

  @Action(UpdateScatterData)
  updateScatterData({getState, patchState, dispatch}: StateContext<ProblemInfoStateModel>): void {
    const state = getState();
    const scatterData = this.getDefaultScatterData();

    const xAxisKey = state.xAxisActive?.key;
    const yAxisKey = state.yAxisActive?.key;

    let xAxisData = state.modelsMetricsTableData.map((row) => (row[xAxisKey] as ITableCell)?.value);
    let yAxisData = state.modelsMetricsTableData.map((row) => (row[yAxisKey] as ITableCell)?.value);
    xAxisData = xAxisData.filter((xAxisDataEntry) => !!xAxisDataEntry);
    yAxisData = yAxisData.filter((yAxisDataEntry) => !!yAxisDataEntry);

    scatterData[0].x = xAxisData;
    scatterData[0].hovertemplate = `${state.xAxisActive?.label}: <b>%{x}</b> ${state.xAxisActive?.unit}`;
    scatterData[0].y = yAxisData;
    scatterData[0].hovertemplate = `${state.yAxisActive?.label}: <b>%{x}</b> ${state.yAxisActive?.unit}` + scatterData[0].hovertemplate;

    patchState({
      scatterData,
    });
    dispatch(new UpdateScatterLayout());
  }

  @Action([UpdateScatterLayout, SetTheme])
  updateScatterLayout({getState, patchState}: StateContext<ProblemInfoStateModel>): void {
    const state = getState();
    const theme = this.store.selectSnapshot(IdlpRootState.theme);

    let scatterLayout = this.getDefaultLayout();
    scatterLayout = this.useTheme(scatterLayout, theme);

    scatterLayout.xaxis.title = state.xAxisActive?.label;
    if (state.xAxisActive?.unit?.trim().length) {
      scatterLayout.xaxis.title = `${scatterLayout.xaxis.title} (${state.xAxisActive?.unit})`;
    }
    scatterLayout.yaxis.title = state.yAxisActive?.label;
    if (state.yAxisActive?.unit?.trim().length) {
      scatterLayout.yaxis.title = `${scatterLayout.yaxis.title} (${state.yAxisActive.unit})`;
    }
    patchState({
      scatterLayout,
    });
  }

  private getColumnsFromModels(models: IModel[]): IModelsMetricsTableColumn[] {
    const columns: IModelsMetricsTableColumn[] = [{key: 'name', label: 'Name'}];
    if (models.length === 0) {
      return columns;
    }
    models.forEach((model: IModel) => {
      if (!('evaluates' in model)) {
        return;
      }
      Object.entries(model?.evaluates || {}).forEach(([_, evaluate]) => {
        if (!evaluate) {
          return;
        }
        if (!evaluate?.metrics?.length) {
          return;
        }
        evaluate.metrics.forEach((metric) => {
          const column = {key: metric.key, label: metric.displayName, unit: metric.unit};
          if (!columns.find((c) => c.key === column.key)) {
            columns.push(column);
          }
        });
      });
    });
    return columns;
  }

  private getDefaultScatterData(): any[] {
    return [{
      x: [],
      y: [],
      hovertemplate: '',
      hoverlabel: {
        namelength: 0
      },
      type: 'scatter',
      mode: 'markers',
      marker: {
        size: 16,
        color: '#1b7ac5',
        line: {
          color: '#444444',
          width: 1
        }
      }
    }];
  }

  private getDefaultLayout(): any {
    return {
      xaxis: {
        title: ''
      },
      yaxis: {
        title: ''
      },
      width: 1280,
      hovermode: 'closest',
      font: {
        family: '"Barlow Semi Condensed", Roboto, "Helvetica Neue", sans-serif',
        size: 16
      },
      margin: {
        t: 35,
        r: 0,
        b: 55,
        l: 55
      }
    };
  }

  private useTheme(layout: any, theme: string): any {
    const COLOR_MAP = {
      textcolorLight: '#333333',
      gridcolorLight: '#dedede',
      zerolinecolorLight: '#dedede',
      bgcolorLight: '#ffffff',

      textcolorDark: '#ffffff',
      gridcolorDark: '#666666',
      zerolinecolorDark: '#666666',
      bgcolorDark: '#333333'
    };
    switch (theme) {
    case environment.themes.light:
      layout = Object.assign(layout, {
        paper_bgcolor: COLOR_MAP.bgcolorLight,
        plot_bgcolor: COLOR_MAP.bgcolorLight,
        font: {
          color: COLOR_MAP.textcolorLight
        },
        xaxis: {
          gridcolor: COLOR_MAP.gridcolorLight,
          zerolinecolor: COLOR_MAP.zerolinecolorLight
        },
        yaxis: {
          gridcolor: COLOR_MAP.gridcolorLight,
          zerolinecolor: COLOR_MAP.zerolinecolorLight
        }
      });
      break;
    case environment.themes.dark:
      layout = Object.assign(layout, {
        paper_bgcolor: COLOR_MAP.bgcolorDark,
        plot_bgcolor: COLOR_MAP.bgcolorDark,
        font: {
          color: COLOR_MAP.textcolorDark
        },
        xaxis: {
          gridcolor: COLOR_MAP.gridcolorDark,
          zerolinecolor: COLOR_MAP.zerolinecolorDark
        },
        yaxis: {
          gridcolor: COLOR_MAP.gridcolorDark,
          zerolinecolor: COLOR_MAP.zerolinecolorDark
        }
      });
      break;
    default:
      return;
    }
    return layout;
  }


}
