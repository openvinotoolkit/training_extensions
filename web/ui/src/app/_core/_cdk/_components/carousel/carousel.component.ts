/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {Component, Input, OnInit} from '@angular/core';

const NO_IMAGE_URL = 'https://via.placeholder.com/400x250?text=No+Image';

@Component({
  selector: 'idlp-cdk-carousel',
  templateUrl: './carousel.component.html',
  styleUrls: ['./carousel.component.scss']
})
export class IdlpCdkCarouselComponent implements OnInit {
  @Input()
  images: string[];

  @Input()
  width = 300;

  @Input()
  height = 200;

  currentImageIndex = 0;

  get currentImage(): string {
    return this.images[this.currentImageIndex];
  }

  ngOnInit(): void {
    for (let i = 0; i <= this.images?.length - 1; i++) {
      this.checkImageExist(this.images[i], (isExist: boolean) => {
        if (!isExist) {
          this.images[i] = null;
        }
      });
    }
  }

  getNextImage(): void {
    if (!this.images || !this.images.length) {
      return;
    }
    if (this.currentImageIndex === this.images.length - 1) {
      this.currentImageIndex = 0;
      return;
    }
    this.currentImageIndex++;
  }

  private checkImageExist(url: string, callback: Function): void {
    const img = new Image();
    img.onload = () => callback(true);
    img.onerror = () => callback(false);
    img.src = url;
  }
}
