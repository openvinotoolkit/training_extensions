/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {Component, EventEmitter, Input, Output, ViewChild} from '@angular/core';
import {MatMenu} from "@angular/material/menu";
import {IModel} from "@idlp/root/models";

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

  @Output()
  onMenuItemClick: EventEmitter<any> = new EventEmitter<any>();

  get isHiddenOnChart(): boolean {
    const hoc: string[] = JSON.parse(localStorage.getItem('hoc'));
    if (hoc) return hoc.includes(this.model.id);
    return this.model.showOnChart;
  }
}
