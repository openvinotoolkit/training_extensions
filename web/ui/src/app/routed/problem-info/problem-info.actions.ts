/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */
import {IAbstractList} from '@idlp/root/models';
import {IBuild, IModel, IModelsMetricsTableColumn, IProblem} from '@idlp/routed/problem-info/problem-info.models';

export class Reset {
  static readonly type = '[Problem Info] Reset';

  constructor() {
  }
}

export class UpdateProblemDetails {
  static readonly type = 'PROBLEM_DETAILS';

  constructor(public data: IProblem) {
  }
}

export class UpdateModels {
  static readonly type = 'MODEL_LIST';

  constructor(public data: IAbstractList<IModel>) {
  }
}

export class ModelDelete {
  static readonly type = 'MODEL_DELETE';

  constructor(public data: any) {
  }
}

export class UpdateBuilds {
  static readonly type = 'BUILD_LIST';

  constructor(public data: IAbstractList<IBuild>) {
  }
}

export class UpdateActiveBuild {
  static readonly type = '[ActiveBuild] Update';

  constructor(public data: IBuild) {
  }
}

export class UpdateProblemId {
  static readonly type = '[ProblemId] Update';

  constructor(public data: string) {
  }
}


export class UpdateModelsMetricsTableData {
  static readonly type = '[ModelsMetricsTableData] Update';

  constructor(public models: IModel[], public columns: IModelsMetricsTableColumn[], public  buildId: string) {
  }
}


export class UpdateModelsMetricsTableColumns {
  static readonly type = '[ModelsMetricsTableColumns] Update';

  constructor(public columns: IModelsMetricsTableColumn[]) {
  }
}

export class UpdateXAxis {
  static readonly type = '[xAxis] Update';

  constructor() {
  }
}


export class UpdateYAxis {
  static readonly type = '[yAxis] Update';

  constructor() {
  }
}


export class UpdateXAxisActive {
  static readonly type = '[xAxisActive] Update';

  constructor(public xAxis: IModelsMetricsTableColumn) {
  }
}

export class UpdateYAxisActive {
  static readonly type = '[yAxisActive] Update';

  constructor(public yAxis: IModelsMetricsTableColumn) {
  }
}


export class UpdateScatterData {
  static readonly type = '[ScatterData] Update';

  constructor() {
  }
}

export class UpdateScatterLayout {
  static readonly type = '[ScatterLayout] Update';

  constructor() {
  }
}
