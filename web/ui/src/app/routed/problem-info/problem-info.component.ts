/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {HttpClient} from '@angular/common/http';
import {Component, OnDestroy, OnInit} from '@angular/core';
import {FormBuilder} from '@angular/forms';
import {MatDialog} from '@angular/material/dialog';
import {MatSnackBar} from '@angular/material/snack-bar';
import {ActivatedRoute} from '@angular/router';
import {environment} from '@environments/environment';
import {IdlpFineTuneDialogComponent} from '@idlp/components/fine-tune-dialog/fine-tune-dialog.component';
import {WebsocketService} from '@idlp/providers/websocket.service';
import {IBuild, IFineTuneDialogData, IModel, IProblem} from '@idlp/root/models';
import {WsConnect} from '@idlp/root/ws.actions';
import {WS} from '@idlp/root/ws.events';
import {WsState} from '@idlp/root/ws.state';
import {
  ModelDelete,
  Reset,
  UpdateActiveBuild,
  UpdateBuilds,
  UpdateModels,
  UpdateProblemDetails,
  UpdateProblemId
} from '@idlp/routed/problem-info/problem-info.actions';
import {ProblemInfoState} from '@idlp/routed/problem-info/problem-info.state';
import {Select, Store} from '@ngxs/store';
import {SendWebSocketMessage} from '@ngxs/websocket-plugin';
import {iif, Observable, Subject, timer} from 'rxjs';
import {delay, first, map, take, takeUntil} from 'rxjs/operators';

@Component({
  selector: 'idlp-problem-info',
  templateUrl: './problem-info.component.html',
  styleUrls: ['./problem-info.component.scss']
})
export class IdlpProblemInfoComponent implements OnInit, OnDestroy {
  @Select(ProblemInfoState.problem) problem$: Observable<IProblem>;
  @Select(ProblemInfoState.models) models$: Observable<IModel[]>;
  @Select(ProblemInfoState.builds) builds$: Observable<IBuild[]>;
  @Select(ProblemInfoState.problemId) problemId$: Observable<string>;
  @Select(ProblemInfoState.activeBuild) activeBuild$: Observable<IBuild>;
  @Select(WsState.connected) connected$: Observable<boolean>;

  problemId: string;
  problem: IProblem;
  models: IModel[] = [];
  activeBuild: IBuild;

  private destroy$: Subject<any> = new Subject();

  constructor(
    private dialogService: MatDialog,
    private fb: FormBuilder,
    private http: HttpClient,
    private route: ActivatedRoute,
    private snackBar: MatSnackBar,
    private store: Store,
    private websocketService: WebsocketService,
  ) {
  }

  ngOnInit(): void {
    this.store.dispatch(new Reset());
    this.store.dispatch(new WsConnect());
    this.store.dispatch(new UpdateProblemId(this.route.snapshot.paramMap.get('id')));
    this.problemId$.subscribe(problemId => this.problemId = problemId);
    this.problem$.subscribe(problem => this.problem = problem);
    this.models$.subscribe(models => this.models = models);
    this.activeBuild$.subscribe(activeBuild => this.activeBuild = activeBuild);
    this.connected$
      .pipe(map(connected => iif(() => connected === true)))
      .subscribe(() => {
        timer(0, 5000)
          .pipe(takeUntil(this.destroy$))
          .subscribe(() => {
            this.store.dispatch(new SendWebSocketMessage({
              event: UpdateModels.type,
              data: {problemId: this.problemId},
            }));
          });
        this.store.dispatch(new SendWebSocketMessage({
          event: UpdateProblemDetails.type,
          data: {id: this.problemId},
        }));

        this.store.dispatch(new SendWebSocketMessage({
          event: UpdateBuilds.type,
          data: {problemId: this.problemId},
        }));
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  buildDisplayFn(build?: IBuild): string {
    return build?.name ?? '';
  }

  setBuild($event: any): void {
    this.store.dispatch(new UpdateActiveBuild($event.option.value));
  }

  menuItemClick(menuTrigger: { [key: string]: any }): void {
    switch (menuTrigger.action) {
    case 'finetune':
      this.fineTuneDialogOpen(menuTrigger.model);
      break;
    case 'evaluate':
      this.evaluateModel(menuTrigger.model);
      break;
    case 'tensorboard':
      this.navigateToTensorBoard(menuTrigger.model);
      break;
    case 'log':
      this.navigateToLogBoard(menuTrigger.model);
      break;
    case 'showonchart':
      this.setShowOnChart(menuTrigger.model);
      break;
    case 'delete':
      this.delete(menuTrigger.model);
      break;
    default:
      return;
    }
  }

  delete(model: IModel): void {
    this.store.dispatch(new SendWebSocketMessage({
      event: ModelDelete.type,
      data: {id: model.id},
    }));
  }

  showModelInTable(modelId: string): void {
    const target = document.getElementById(modelId);
    if (!target) {
      return;
    }
    target.classList.add('highlight');
    target.scrollIntoView();
    timer(1200)
      .pipe(take(1))
      .subscribe(() => {
        target.classList.remove('highlight');
      });
  }

  private navigateToTensorBoard(model: IModel): void {
    const tensorBoardUri = `${environment.tensorBoardUrl}/run?folder=${model.tensorBoardLogDir}`;
    this.http.get(tensorBoardUri)
      .pipe(
        first(),
        delay(2000)
      )
      .subscribe((): any => window.open(environment.tensorBoardUrl, '_blank'));
  }

  private navigateToLogBoard(model: IModel): void {
    window.open(`${environment.filebrowserUrl}/files/idlp${model.dirPath}/output.log`, '_blank');
  }

  private evaluateModel(model: IModel): void {
    this.websocketService.send(
      WS.SEND.MODEL_EVALUATE,
      {
        modelId: model.id,
        buildId: this.activeBuild.id,
        problemId: this.problemId
      }
    );
    this.snackBar.open(`Evaluation "${model.name}" has been started`, 'OK', {duration: 5000});
  }

  private setShowOnChart(model: IModel): void {
    if (!model?.id) {
      return;
    }
    if (!localStorage.getItem('hoc')) {
      localStorage.setItem('hoc', JSON.stringify([]));
    }

    const hoc: string[] = JSON.parse(localStorage.getItem('hoc'));
    if (hoc.includes(model.id)) {
      hoc.splice(hoc.indexOf(model.id), 1);
    } else {
      hoc.push(model.id);
    }
    localStorage.setItem('hoc', JSON.stringify(hoc));
    this.websocketService.send(WS.SEND.BUILD_LIST, {problemId: this.problemId});
  }

  private fineTuneDialogOpen(model: IModel): void {
    this.dialogService.open(IdlpFineTuneDialogComponent, {
      autoFocus: false,
      disableClose: true,
      width: '550px',
      data: {
        problemId: this.problemId,
        problemName: this.problem.title,
        buildId: this.activeBuild.id,
        model
      }
    })
      .afterClosed()
      .pipe(take(1))
      .subscribe((fineTuneDialogData: IFineTuneDialogData) => {
        if (fineTuneDialogData && fineTuneDialogData.build?.id) {
          this.websocketService.send(WS.SEND.MODEL_FINE_TUNE, {
            problemId: this.problemId,
            parentModelId: model.id,
            buildId: fineTuneDialogData.build.id,
            epochs: fineTuneDialogData.epochs,
            name: fineTuneDialogData.name,
            saveAnnotatedValImages: fineTuneDialogData.saveAnnotatedValImages,
            batchSize: fineTuneDialogData.advanced ? fineTuneDialogData.batchSize : undefined,
            gpuNumber: fineTuneDialogData.advanced ? fineTuneDialogData.gpuNumber : undefined
          });
        }
      });
  }
}
