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

interface IBaseEntity {
  id: string;
}

export interface IAbstractList<T> {
  total: number;
  items: T[];
}

export interface IProblem extends IBaseEntity {
  title: string;
  subtitle: string;
  description: string;
  type: string;
  imagesUrls?: string[];
}

export interface IAssetProgress {
  total: number;
  done: number;
  percentage: number;
}

export interface IAsset extends IBaseEntity {
  problemId: string;
  id: string;
  name: string;
  parentFolder: string;
  status: string;
  type: string;
  cvatId: number;
  url: string;
  progress: IAssetProgress;
  buildSplit: IAssetBuildSplit;
}

export interface IBuild extends IBaseEntity {
  problemId: string;
  id: string;
  name: string;
  status: string;
}

export interface IFineTuneDialogData {
  name: string;
  build: IBuild;
  epochs: number;
  batchSize: number;
  gpuNumber: number;
  advanced: boolean;
  saveAnnotatedValImages: boolean;
}

export interface IProblemCreateDialogData {
  title: string;
  subtitle: string;
  labels: any;
  description: string;
  image: File;
}

export interface IAssetBuildSplit {
  train: number;
  val: number;
  test: number;
}

export interface IAssetBuildState {
  id: string;
  train: number;
  val: number;
  test: number;
}
