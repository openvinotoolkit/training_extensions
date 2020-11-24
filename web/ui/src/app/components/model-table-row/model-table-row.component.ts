import {Component, Input} from '@angular/core';


@Component({
  selector: 'idlp-model-table-row-status',
  templateUrl: './model-table-row.component.html',
  styleUrls: ['./model-table-row.component.scss']
})
export class IdlpModelTableRowComponent {
  @Input()
  showProgress = false;

  @Input()
  message = '';
}
