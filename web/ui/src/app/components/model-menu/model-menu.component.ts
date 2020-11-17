/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {Component, EventEmitter, Input, Output, ViewChild} from '@angular/core';
import {MatMenu} from '@angular/material/menu';
import {IModel} from '@idlp/routed/problem-info/problem-info.models';


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
  model: IModel;

  @Input()
  isBuildValidForEvaluate: boolean;

  @Output()
  onMenuItemClick: EventEmitter<any> = new EventEmitter<any>();

  get isHiddenOnChart(): boolean {
    const hoc: string[] = JSON.parse(localStorage.getItem('hoc'));
    if (hoc) return hoc.includes(this.model.id);
    return this.model.showOnChart;
  }

  get isFineTuneDisabled(): boolean {
    return ['trainInProgress', 'trainFailed'].includes(this.model?.status);
  }

  get isTensorboardDisabled(): boolean {
    return ['trainDefault', 'evaluateDefault'].includes(this.model?.status);
  }

  get isLogDisabled(): boolean {
    return ['trainDefault', 'evaluateDefault'].includes(this.model?.status);
  }

  get isShowOnChartDisabled(): boolean {
    return ['inProgress', 'initiate'].includes(this.model?.status);
  }

  get isEvaluateDisabled(): boolean {
    return ['trainInProgress', 'trainFailed', 'evaluateDefault', 'evaluateInProgress', 'evaluateFinished', 'evaluateFailed'].includes(this.model?.status)
  }

  get isDeleteDisabled(): boolean {
    return ['trainDefault', 'evaluateDefault'].includes(this.model?.status);
  }
}
