/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {IAbstractList} from '@idlp/root/models';
import {IProblem} from '@idlp/routed/problem-list/problem-list.models';

export class ProblemList {
  static readonly type = 'PROBLEM_LIST';

  constructor(public data: IAbstractList<IProblem>) {
  }
}

export class ProblemCreate {
  static readonly type = 'PROBLEM_CREATE';

  constructor(public problem: IProblem) {
  }
}

export class ProblemDelete {
  static readonly type = 'PROBLEM_DELETE';

  constructor(public data: any) {
  }
}
