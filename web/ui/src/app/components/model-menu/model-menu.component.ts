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
import {Component, EventEmitter, Input, Output, ViewChild} from '@angular/core';
import {MatMenu} from '@angular/material/menu';
import {IModelTableRow} from '@idlp/routed/problem-info/problem-info.models';


@Component({
  selector: 'idlp-model-menu',
  templateUrl: './model-menu.component.html',
  styleUrls: ['./model-menu.component.scss'],
  exportAs: 'modelMenu'
})
export class IdlpModelMenuComponent {
  @ViewChild(MatMenu, {static: true})
  instance: MatMenu;

  @Input()
  modelRow: IModelTableRow;

  @Input()
  isBuildValidForEvaluate: boolean;

  @Output()
  onMenuItemClick: EventEmitter<any> = new EventEmitter<any>();

  get isHiddenOnChart(): boolean {
    const hoc: string[] = JSON.parse(localStorage.getItem('hoc'));
    if (hoc) return hoc.includes(this.modelRow.id);
    return this.modelRow.showOnChart;
  }

  get isFineTuneDisabled(): boolean {
    return ['trainInProgress', 'trainFailed'].includes(this.modelRow?.trainStatus);
  }

  get isTensorboardDisabled(): boolean {
    return this.modelRow?.trainStatus === 'trainDefault' || this.modelRow?.evalStatus === 'evaluateDefault';
  }

  get isLogDisabled(): boolean {
    return this.modelRow?.trainStatus === 'trainDefault' || this.modelRow?.evalStatus === 'evaluateDefault';
  }

  get isEvaluateDisabled(): boolean {
    const trainStatuses = ['trainInProgress', 'trainFailed'];
    const evalStatuses = ['evaluateDefault', 'evaluateInProgress', 'evaluateFinished', 'evaluateFailed'];
    return trainStatuses.includes(this.modelRow?.trainStatus) || evalStatuses.includes(this.modelRow?.evalStatus);
  }

  get isDeleteDisabled(): boolean {
    return this.modelRow?.trainStatus === 'trainDefault' || this.modelRow?.evalStatus === 'evaluateDefault';
  }
}
