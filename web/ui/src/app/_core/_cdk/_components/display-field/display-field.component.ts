/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {Component, Input} from '@angular/core';

@Component({
  selector: 'idlp-display-field',
  templateUrl: './display-field.component.html',
  styleUrls: ['./display-field.component.scss']
})
export class IdlpDisplayFieldComponent {
  @Input()
  label: string;

  @Input()
  value: string;

  @Input()
  hAlign: string;
}
