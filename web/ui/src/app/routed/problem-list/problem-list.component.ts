/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {Component, OnDestroy, OnInit} from '@angular/core';
import {MatDialog} from '@angular/material/dialog';
import {Router} from '@angular/router';
import {IdlpProblemCreateDialogComponent} from '@idlp/components/problem-create-dialog/problem-create-dialog.component';
import {IProblem, IProblemCreateDialogData} from '@idlp/root/models';
import {WsConnect} from '@idlp/root/ws.actions';
import {WsState} from '@idlp/root/ws.state';
import {ProblemCreate, ProblemList} from '@idlp/routed/problem-list/problem-list.actions';
import {ProblemListState} from '@idlp/routed/problem-list/problem-list.state';
import {Select, Store} from '@ngxs/store';
import {SendWebSocketMessage} from '@ngxs/websocket-plugin';
import {iif, Observable, Subject} from 'rxjs';
import {map, take} from 'rxjs/operators';

@Component({
  selector: 'idlp-problem-list',
  templateUrl: './problem-list.component.html',
  styleUrls: ['./problem-list.component.scss']
})
export class IdlpProblemListComponent implements OnInit, OnDestroy {
  @Select(ProblemListState.getProblemList) problems$: Observable<IProblem[]>;
  @Select(WsState.connected) connected$: Observable<boolean>;

  private destroy$: Subject<any> = new Subject();

  constructor(
    private dialogService: MatDialog,
    private router: Router,
    private store: Store,
  ) {

  }

  ngOnInit(): void {
    this.store.dispatch(new WsConnect());
    this.connected$
      .pipe(map(connected => iif(() => connected === true)))
      .subscribe((_) => {
        this.store.dispatch(new SendWebSocketMessage({
          event: ProblemList.type,
        }));
      });

  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  itemClick(problem: IProblem): void {
    if (problem.type !== 'generic') {
      this.router.navigate([`/${problem.id}`]);
      return;
    }
    this.dialogService.open(IdlpProblemCreateDialogComponent, {
      autoFocus: false,
      disableClose: true,
      width: '550px'
    })
      .afterClosed()
      .pipe(take(1))
      .subscribe((problemCreateDialogData: IProblemCreateDialogData) => {
        if (problemCreateDialogData) {
          const labels: string = problemCreateDialogData.labels?.trim();
          if (labels?.length) {
            const labelsArray = [];
            labels.split(',').map((label: string) => {
              if (label?.trim().length) {
                labelsArray.push({name: label.trim(), arguments: []});
              }
            });
            problemCreateDialogData.labels = labelsArray;
          }
          this.store.dispatch(new SendWebSocketMessage({
            event: ProblemCreate.type,
            data: problemCreateDialogData
          }));
        }
      });
  }
}
