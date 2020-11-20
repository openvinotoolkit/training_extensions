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

import {Component, EventEmitter, Input, OnInit, Output} from '@angular/core';
import {IAsset, IAssetBuildState} from '@idlp/root/models';
import {MatCheckboxChange} from '@angular/material/checkbox';

@Component({
  selector: 'idlp-asset-card-item',
  templateUrl: './asset-card-item.component.html',
  styleUrls: ['./asset-card-item.component.scss']
})
export class IdlpAssetCardItemComponent implements OnInit {
  @Input()
  asset: IAsset;

  @Input()
  width: string | number = 150;

  @Input()
  height: string | number = 'auto';

  @Output()
  onClick: EventEmitter<IAsset> = new EventEmitter<IAsset>();

  @Output()
  onPull: EventEmitter<IAsset> = new EventEmitter<IAsset>();

  @Output()
  onPush: EventEmitter<IAsset> = new EventEmitter<IAsset>();

  @Output()
  onAssign: EventEmitter<IAssetBuildState> = new EventEmitter<IAssetBuildState>();

  train: boolean;
  val: boolean;

  get pushDisabled(): boolean {
    return this.asset.status === 'initialPullReady' || this.asset.status === 'pullReady' || this.asset.status === 'pullInProgress';
  }

  get pullDisabled(): boolean {
    return this.asset.status === 'initial' || this.asset.status === 'pushInProgress';
  }

  get showPushAction(): boolean {
    return this.asset.status !== 'pushInProgress';
  }

  get showPullAction(): boolean {
    return this.asset.status !== 'pullInProgress';
  }

  get pushActionTooltip(): string {
    switch (this.asset.status) {
    case 'initial':
      return 'Push to CVAT';
    case 'pushInProgress':
      return 'Pushing to CVAT';
    }
    return null;
  }

  get pullActionTooltip(): string {
    switch (this.asset.status) {
    case 'pullReady':
      return 'Pull from CVAT';
    case 'pullInProgress':
      return 'Pulling from CVAT';
    }
    return null;
  }

  get isClickAllowed(): boolean {
    return this.asset.type === 'folder' || this.asset.url?.trim().length > 0;
  }

  ngOnInit(): void {
    this.train = Boolean(this.asset?.buildSplit.train);
    this.val = Boolean(this.asset?.buildSplit.val);
  }

  assign(event: MatCheckboxChange): void {
    if (event.checked) {
      switch (event.source.name) {
      case 'train':
        this.val = false;
        break;
      case 'val':
        this.train = false;
        break
      }
    }
    this.onAssign.emit({
      id: this.asset.id,
      train: Number(this.train),
      val: Number(this.val),
      test: Number(this.val)
    });
  }
}
