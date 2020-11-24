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

import {Component, ElementRef, ViewChild} from '@angular/core';
import {FormBuilder, FormGroup, Validators} from '@angular/forms';
import {MatSnackBar} from "@angular/material/snack-bar";

const IMAGE_MAX_SIZE = 2097152; // 2Mb
const IMAGE_MAX_WIDTH = 1024;  // 1024px
const IMAGE_MAX_HEIGHT = 768; // 768px

@Component({
  selector: 'idlp-problem-create-dialog',
  templateUrl: './problem-create-dialog.component.html',
  styleUrls: ['./problem-create-dialog.component.scss']
})
export class IdlpProblemCreateDialogComponent {
  @ViewChild('fileInput', {static: false})
  fileInput: ElementRef;

  imageSrc: string;
  form: FormGroup;

  constructor(
    private fb: FormBuilder,
    private snackBar: MatSnackBar,
  ) {
    this.form = this.fb.group({
      title: ['', Validators.required],
      subtitle: ['', Validators.required],
      labels: ['', Validators.required],
      description: ['', Validators.required],
      image: null
    });
  }

  get labels(): string {
    return this.form.get('labels').value;
  }

  get isLabelsFormatValid(): boolean {
    const regex = /((,*\s*)*\w+(,*\s*)*)+/;
    return regex.test(this.labels.trim());
  }

  selectImage(): void {
    const fileInput = this.fileInput.nativeElement;
    const reader = new FileReader();
    reader.onload = this.handleReaderLoad.bind(this);
    fileInput.onchange = () => {
      const file = fileInput.files[0];
      if (file) {
        if (file.size > IMAGE_MAX_SIZE) {
          this.snackBar.open(
            ` Image size limit is ${Math.round(IMAGE_MAX_SIZE / 1024 / 1024)}Mb.
             Actual image size is ${Math.round(file.size / 1024 / 1024)}Mb`,
            'OK',
            {duration: 5000}
          );
          this.clearImage();
          return;
        }
        reader.readAsDataURL(file);
      }
    };
    fileInput.click();
  }

  clearImage(): void {
    const fileInput = this.fileInput.nativeElement;
    fileInput.value = null;
    this.imageSrc = null;
    this.form.get('image').setValue(null);
  }

  private handleReaderLoad(event) {
    const reader = event.target;
    const image = new Image();
    image.src = reader.result;
    image.onload = rs => {
      const imgWidth = rs.currentTarget['width'];
      const imgHeight = rs.currentTarget['height'];
      if (imgWidth > IMAGE_MAX_WIDTH && imgHeight > IMAGE_MAX_HEIGHT) {
        this.snackBar.open(`Maximum image dimensions allowed ${IMAGE_MAX_WIDTH}x${IMAGE_MAX_HEIGHT}px`, 'OK', {duration: 5000})
        this.clearImage();
      } else if (imgWidth > IMAGE_MAX_WIDTH) {
        this.snackBar.open(`Maximum image width allowed is ${IMAGE_MAX_WIDTH}px`, 'OK', {duration: 5000})
        this.clearImage();
      } else if (imgHeight > IMAGE_MAX_HEIGHT) {
        this.snackBar.open(`Maximum image height allowed is ${IMAGE_MAX_HEIGHT}px`, 'OK', {duration: 5000})
        this.clearImage();
      } else {
        this.imageSrc = image.src;
        this.form.get('image').setValue(this.imageSrc);
      }
    }
  }
}
