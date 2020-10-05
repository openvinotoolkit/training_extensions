/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */
import { Pipe, PipeTransform } from '@angular/core';
import { Utils } from '@idlp/utils/utils';

const DEFAULT_REPLACEMENT = '\u2014';

@Pipe({
  name: 'dashed'
})
export class DashedPipe implements PipeTransform {
  transform(value: any, replacement?: string): string {
    return Utils.isEmpty(value) ? replacement ?? DEFAULT_REPLACEMENT : value;
  }
}
