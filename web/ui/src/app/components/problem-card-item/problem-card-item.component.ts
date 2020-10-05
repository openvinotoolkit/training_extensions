/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {Component, EventEmitter, Input, Output} from '@angular/core';
import {IProblem} from '@idlp/root/models';

const DESCRIPTION_LENGTH = 235;

@Component({
  selector: 'idlp-problem-card-item',
  templateUrl: './problem-card-item.component.html',
  styleUrls: ['./problem-card-item.component.scss']
})
export class IdlpProblemCardItemComponent {
  @Input()
  width: number;

  @Input()
  height: number;

  @Input()
  problem: IProblem;

  @Output()
  enter: EventEmitter<void> = new EventEmitter<void>();

  get description(): string {
    if (this.problem.description) {
      if (this.problem.description.length > DESCRIPTION_LENGTH) {
        return `${this.problem.description.substr(0, DESCRIPTION_LENGTH)}...`;
      }
      return this.problem.description;
    }
    return '';
  }
}
