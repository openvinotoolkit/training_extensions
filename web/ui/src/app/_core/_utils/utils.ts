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

export class Utils {
  static isEmpty(obj: any): boolean {
    return obj === null || obj === undefined
      || (typeof obj === 'string' && obj.trim().length === 0)
      || (typeof obj === 'object' && Object.entries(obj).length === 0);
  }

  static objectArraySort<T>(objects: T[], sortProperty: string, direction: 'asc' | 'desc' = 'asc') {
    direction = (direction === 'asc' || direction === 'desc') ? direction : 'asc';
    return objects.sort((metricA: T, metricB: T) => {
      const a = metricA[sortProperty].toLowerCase();
      const b = metricB[sortProperty].toLowerCase();
      if (a < b) {
        return direction === 'asc' ? -1 : 1;
      }
      if (a > b) {
        return direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
  }

  static twoDigits(dateEntry: number): string {
    return dateEntry < 10 ? `0${dateEntry}` : `${dateEntry}`;
  }
}
