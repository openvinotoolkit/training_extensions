/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {Component, Inject, OnDestroy, OnInit} from '@angular/core';
import {FormBuilder, FormGroup, Validators} from '@angular/forms';
import {MAT_DIALOG_DATA} from '@angular/material/dialog';
import {IAbstractList, IBuild, IModel} from '@idlp/root/models';
import {Observable, of, Subject} from 'rxjs';
import {filter, startWith, takeUntil} from 'rxjs/operators';
import {WS} from '@idlp/root/ws.events';
import {WebsocketService} from '@idlp/providers/websocket.service';

@Component({
  selector: 'idlp-fine-tune-dialog',
  templateUrl: './fine-tune-dialog.component.html',
  styleUrls: ['./fine-tune-dialog.component.scss']
})
export class IdlpFineTuneDialogComponent implements OnInit, OnDestroy {
  builds: IBuild[] = [];
  filteredBuilds: Observable<IBuild[]> = of([]);

  buildId: string;
  model: IModel;

  form: FormGroup;

  private destroy$: Subject<any> = new Subject();

  constructor(
    private fb: FormBuilder,
    private websocketService: WebsocketService,
    @Inject(MAT_DIALOG_DATA) public data: any
  ) {
    this.buildId = data.buildId;
    this.model = data.model;

    this.websocketService
      .on<IAbstractList<IBuild>>(WS.ON.BUILD_LIST)
      .pipe(
        takeUntil(this.destroy$),
        filter((builds: IAbstractList<IBuild>) => builds?.items?.length > 0)
      )
      .subscribe((builds: IAbstractList<IBuild>) => {
        this.builds = builds.items.filter((build: IBuild) => build.status !== 'tmp' && build.name !== 'default');
        this.builds = this.builds.sort((buildA: IBuild, buildB: IBuild) => buildA.name?.trim().toLocaleLowerCase() > buildB.name?.trim().toLocaleLowerCase() ? -1 : 1);
        this.filteredBuilds = of(this.builds);
      });

    this.form = this.fb.group({
      batchSize: 1,
      gpuNumber: 1,
      saveAnnotatedValImages: [false, Validators.required],
      advanced: [false, Validators.required],
      name: [this.getDefaultModelName(data.problemName), Validators.required],
      build: [null, Validators.required],
      epochs: [5, [Validators.required, Validators.min(1), Validators.max(100)]]
    });
  }

  get formValid(): boolean {
    return this.form.valid && (this.form.get('build').value instanceof Object);
  }

  get advancedEnabled(): boolean {
    return this.form.get('advanced').value;
  }

  ngOnInit(): void {
    this.websocketService.send(WS.SEND.BUILD_LIST, {page: 1, size: 500});

    this.form.get('advanced').valueChanges
      .pipe(
        takeUntil(this.destroy$)
      )
      .subscribe((enable: any) => {
        if (enable) {
          this.form.get('batchSize').setValidators([Validators.required, Validators.min(1), Validators.max(1000)]);
          this.form.get('gpuNumber').setValidators([Validators.required, Validators.min(1), Validators.max(100)]);
        } else {
          this.form.get('batchSize').clearValidators();
          this.form.get('gpuNumber').clearValidators();
        }
        this.form.get('batchSize').updateValueAndValidity();
        this.form.get('gpuNumber').updateValueAndValidity();
      });

    this.form.get('build').valueChanges
      .pipe(
        startWith(''),
        takeUntil(this.destroy$)
      )
      .subscribe((filterValue: any) => {
        if (typeof filterValue !== 'string') {
          return;
        }
        this.filteredBuilds = of(this.builds?.filter((build: IBuild) =>
          build.name?.trim().toLowerCase().includes(filterValue.trim().toLowerCase())
        ));
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  buildDisplayFn(build?: IBuild): string {
    return build?.name ?? '';
  }

  getMetricAlign(metricIndex: number): string {
    if (metricIndex > 2) {
      return this.getMetricAlign(metricIndex - 3);
    }
    return metricIndex === 0 ? 'flex-start' : metricIndex === 1 ? 'center' : 'flex-end';
  }

  private getDefaultModelName(problemName = ''): string {
    return `${problemName.toLocaleLowerCase().split(' ').join('-')}-${Date.now()}`;
  }
}
