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

import {Component, EventEmitter, OnDestroy, OnInit, Output} from '@angular/core';
import {FormBuilder, FormGroup} from '@angular/forms';
import {
  UpdateScatterLayout,
  UpdateXAxisActive,
  UpdateYAxisActive
} from '@idlp/routed/problem-info/problem-info.actions';
import {IModel, IModelsMetricsTableColumn} from '@idlp/routed/problem-info/problem-info.models';
import {ProblemInfoState} from '@idlp/routed/problem-info/problem-info.state';
import {Select, Store} from '@ngxs/store';
import {Observable, Subject} from 'rxjs';


@Component({
  selector: 'idlp-scatter-chart',
  templateUrl: './scatter-chart.component.html',
  styleUrls: ['./scatter-chart.component.scss']
})
export class IdlpScatterChartComponent implements OnInit, OnDestroy {
  @Select(ProblemInfoState.xAxis) xAxis$: Observable<IModelsMetricsTableColumn[]>;
  @Select(ProblemInfoState.yAxis) yAxis$: Observable<IModelsMetricsTableColumn[]>;
  @Select(ProblemInfoState.xAxisActive) xAxisActive$: Observable<IModelsMetricsTableColumn>;
  @Select(ProblemInfoState.yAxisActive) yAxisActive$: Observable<IModelsMetricsTableColumn>;
  @Select(ProblemInfoState.scatterData) scatterData$: Observable<any>;
  @Select(ProblemInfoState.scatterLayout) scatterLayout$: Observable<any>;
  @Select(ProblemInfoState.models) modelsSnapshot$: Observable<IModel[]>;

  @Output()
  onClick: EventEmitter<string> = new EventEmitter<string>();

  xAxis: IModelsMetricsTableColumn[] = [];
  yAxis: IModelsMetricsTableColumn[] = [];

  data = [];

  layout = {};

  form: FormGroup;

  private modelsSnapshot: IModel[] = [];

  private destroy$: Subject<any> = new Subject();

  constructor(
    private store: Store,
    private fb: FormBuilder,
  ) {
    this.form = this.fb.group({xAxisMetric: null, yAxisMetric: null});
  }

  ngOnInit(): void {
    this.store.dispatch(new UpdateScatterLayout());
    this.xAxis$.subscribe(xAxis => this.xAxis = xAxis);
    this.yAxis$.subscribe(yAxis => this.yAxis = yAxis);
    this.xAxisActive$.subscribe(xAxisActive => this.form.get('xAxisMetric').setValue(xAxisActive?.key));
    this.yAxisActive$.subscribe(yAxisActive => this.form.get('yAxisMetric').setValue(yAxisActive?.key));
    this.scatterData$.subscribe((data) => this.data = data);
    this.scatterLayout$.subscribe((layout) => this.layout = layout);
    this.modelsSnapshot$.subscribe((models: IModel[]) => this.modelsSnapshot = models);

  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  onChartItemClick($event): void {
    const pointIndex = $event.points[0].pointIndex;
    const model: IModel = this.modelsSnapshot[pointIndex];
    this.onClick.emit(model.id);
  }

  setXAxis($event: any): void {
    const xAxis = this.xAxis.find((v) => v.key === $event.value);
    this.store.dispatch(new UpdateXAxisActive(xAxis));
  }

  setYAxis($event: any): void {
    const yAxis = this.yAxis.find((v) => v.key === $event.value);
    this.store.dispatch(new UpdateYAxisActive(yAxis));
  }
}
