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

import {Component, OnInit} from '@angular/core';
import {WebsocketService} from '@idlp/providers/websocket.service';
import {IAbstractList, IAsset, IAssetBuildState} from '@idlp/root/models';
import {Subject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';
import {ActivatedRoute} from '@angular/router';
import {HttpClient} from '@angular/common/http';
import {WS} from '@idlp/root/ws.events';
import {environment} from '@environments/environment';
import {MatSnackBar} from '@angular/material/snack-bar';
import {Utils} from '@idlp/utils/utils';

@Component({
  selector: 'idlp-problem-assets',
  templateUrl: './problem-assets.component.html',
  styleUrls: ['./problem-assets.component.scss']
})
export class IdlpProblemAssetsComponent implements OnInit {
  problemId: string;

  assets: IAsset[] = [];
  breadcrumbs: IAsset[] = [];

  private destroy$: Subject<any> = new Subject();

  constructor(
    private route: ActivatedRoute,
    private http: HttpClient,
    private snackBar: MatSnackBar,
    private websocketService: WebsocketService
  ) {
    this.websocketService
      .on<IAbstractList<IAsset>>(WS.ON.ASSET_FIND_IN_FOLDER)
      .pipe(
        takeUntil(this.destroy$)
      )
      .subscribe((data: IAbstractList<IAsset>) => this.assets = data.items || []);

    this.websocketService
      .on<string>(WS.ON.BUILD_CREATE)
      .pipe(
        takeUntil(this.destroy$)
      )
      .subscribe(() => {
        this.snackBar.open('Build has been created', 'OK', {duration: 5000});
      });

    this.websocketService
      .on<IAssetBuildState>(WS.ON.BUILD_UPDATE_ASSET_STATE)
      .pipe(
        takeUntil(this.destroy$)
      )
      .subscribe((data: IAssetBuildState) => {
        const assetIndex = this.assets.findIndex((asset: IAsset) => asset.id === data.id);
        this.assets[assetIndex].buildSplit = {train: data.train, val: data.val, test: data.test};
      });

    this.websocketService
      .on<IAsset>(WS.ON.ASSET_SETUP_TO_CVAT)
      .pipe(
        takeUntil(this.destroy$)
      )
      .subscribe((data: IAsset) => {
        const assetIndex = this.assets.findIndex((asset: IAsset) => asset.id === data.id);
        this.assets[assetIndex] = data;
        switch (data.status) {
        case 'pushInProgress':
          this.snackBar.open(`"${data.name}" started to push to CVAT`, 'OK', {duration: 5000});
          break;
        case 'pullReady':
          this.snackBar.open(`"${data.name}" completely pushed to CVAT`, 'OK', {duration: 5000});
          break;
        }
      });

    this.websocketService
      .on<IAsset>(WS.ON.ASSET_DUMP_ANNOTATION)
      .pipe(
        takeUntil(this.destroy$)
      )
      .subscribe((data: IAsset) => {
        const assetIndex = this.assets.findIndex((asset: IAsset) => asset.id === data.id);
        this.assets[assetIndex] = data;
        switch (data.status) {
        case 'pullInProgress':
          this.snackBar.open(`"${data.name}" started to pull from CVAT`, 'OK', {duration: 5000});
          break;
        case 'pullReady':
          this.snackBar.open(`"${data.name}" completely pulled from CVAT`, 'OK', {duration: 5000});
          break;
        }
      });
  }

  get isBuildEnabled(): boolean {
    const selectedAssets: IAsset[] = this.assets.filter((asset: IAsset) => asset.buildSplit.train > 0 || asset.buildSplit.val > 0 || asset.buildSplit.test > 0);
    if (!selectedAssets.length) {
      return false;
    }
    return selectedAssets.every((asset: IAsset) => asset.status === 'pullReady');
  }

  get buildButtonTooltip(): string {
    return this.isBuildEnabled ? 'Create build' : 'Check your assets selection';
  }

  ngOnInit(): void {
    this.problemId = this.route.snapshot.paramMap.get('id');
    this.websocketService.send(WS.SEND.ASSET_FIND_IN_FOLDER, {problemId: this.problemId, root: '.'});
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadAssets(asset?: IAsset): void {
    let root = '.';
    if (!asset) {
      this.breadcrumbs = [];
    } else {
      if (this.breadcrumbs.length && this.breadcrumbs.indexOf(asset) === this.breadcrumbs.length - 1) {
        return;
      }
      if (asset.type !== 'folder') {
        window.open(`${environment.cvatUrl}/tasks/${asset.cvatId}`, '_blank');
        return;
      }
      root = `${asset.parentFolder}/${asset.name}`;
      if (this.breadcrumbs.includes(asset)) {
        this.breadcrumbs.splice(this.breadcrumbs.indexOf(asset) + 1);
      } else {
        this.breadcrumbs.push(asset);
      }
    }
    this.websocketService.send(WS.SEND.ASSET_FIND_IN_FOLDER, {problemId: this.problemId, root});
  }

  setupTaskToCvat(asset: IAsset): void {
    this.websocketService.send(WS.SEND.ASSET_SETUP_TO_CVAT, {id: asset.id});
  }

  downloadTaskFromCvat(asset: IAsset): void {
    this.websocketService.send(WS.SEND.ASSET_DUMP_ANNOTATION, {id: asset.id});
  }

  assetAssign(assetBuildState: IAssetBuildState): void {
    this.websocketService.send(WS.SEND.BUILD_UPDATE_ASSET_STATE, assetBuildState);
  }

  createBuild(): void {
    const currentDate = new Date();
    const year = currentDate.getFullYear();
    const month = Utils.twoDigits(currentDate.getMonth() + 1);
    const date = Utils.twoDigits(currentDate.getDate());
    const hours = Utils.twoDigits(currentDate.getHours());
    const minutes = Utils.twoDigits(currentDate.getMinutes());
    const seconds = Utils.twoDigits(currentDate.getSeconds());
    this.websocketService.send(WS.SEND.BUILD_CREATE, {
      problemId: this.problemId,
      name: `${year}.${month}.${date} ${hours}:${minutes}:${seconds}`
    });
  }
}
