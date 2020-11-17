/**
 * @overview
 * @copyright (c) JSC Intel A/O
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
