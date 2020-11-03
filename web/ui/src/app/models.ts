/**
 * @overview
 * @copyright (c) JSC Intel A/O
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
