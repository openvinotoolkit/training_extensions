/**
 * @overview Utils class
 * @copyright (c) JSC Intel A/O
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
      if (a < b) {return direction === 'asc' ? -1 : 1; }
      if (a > b) {return direction === 'asc' ? 1 : -1; }
      return 0;
    });
  }

  static twoDigits(dateEntry: number): string {
    return dateEntry < 10 ? `0${dateEntry}` : `${dateEntry}`;
  }
}
