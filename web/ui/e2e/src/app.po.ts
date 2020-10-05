/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import {browser, by, element, ElementArrayFinder, ElementFinder} from 'protractor';

export class AppPage {
  get toolbarTitle(): Promise<string> {
    return element(by.css('idlp-root mat-toolbar .title')).getText() as Promise<string>;
  }

  get faceDetectionProblem(): ElementFinder {
    return element.all(by.css('idlp-problem-card-item')).first();
  }

  get faceDetectionProblemTitle(): Promise<string> {
    return element(by.css('idlp-problem-info .title')).getText() as Promise<string>;
  }

  get infoNavigationItem(): ElementFinder {
    return element.all(by.css('idlp-root mat-toolbar .navigation-item')).get(0);
  }

  get assetsNavigationItem(): ElementFinder {
    return element.all(by.css('idlp-root mat-toolbar .navigation-item')).get(1);
  }

  get assetCardItems(): ElementArrayFinder {
    return element.all(by.tagName('idlp-asset-card-item'));
  }

  get createBuildButton(): ElementFinder {
    return element(by.css('.action-build')).element(by.tagName('button'));
  }

  get buildAutocomplete(): ElementFinder {
    return element(by.css('input'));
  }

  get buildsAutocompleteItems(): ElementArrayFinder {
    return element.all(by.tagName('mat-option'));
  }

  get firstAssetCardItem(): ElementFinder {
    return element.all(by.tagName('idlp-asset-card-item')).get(4);
  }

  get secondAssetCardItem(): ElementFinder {
    return element.all(by.tagName('idlp-asset-card-item')).get(5);
  }

  get firstAssetCardItemPushButton(): ElementFinder {
    return this.firstAssetCardItem.element(by.css('.push'));
  }

  get secondAssetCardItemPushButton(): ElementFinder {
    return this.secondAssetCardItem.element(by.css('.push'));
  }

  get firstAssetCardItemLoader(): ElementFinder {
    return this.firstAssetCardItem.element(by.tagName('mat-spinner'));
  }

  get firstAssetCardItemPullButton(): ElementFinder {
    return this.firstAssetCardItem.element(by.css('.pull'));
  }

  get secondAssetCardItemLoader(): ElementFinder {
    return this.secondAssetCardItem.element(by.tagName('mat-spinner'));
  }

  get secondAssetCardItemPullButton(): ElementFinder {
    return this.secondAssetCardItem.element(by.css('.pull'));
  }

  get firstAssetCardItemTrainCheckbox(): ElementFinder {
    return this.firstAssetCardItem.element(by.css('mat-checkbox[name="train"]'));
  }

  get secondAssetCardItemValCheckbox(): ElementFinder {
    return this.secondAssetCardItem.element(by.css('mat-checkbox[name="val"]'));
  }

  get firstAssetCardItemActionWrapper(): ElementFinder {
    return this.firstAssetCardItem.element(by.css('.action-wrapper'));
  }

  get secondAssetCardItemActionWrapper(): ElementFinder {
    return this.secondAssetCardItem.element(by.css('.action-wrapper'));
  }

  get fineTuneModel(): ElementFinder {
    return element.all(by.tagName('tr')).get(2);
  }

  get contextMenu(): ElementFinder {
    return element(by.css('.mat-menu-panel'));
  }

  get fineTuneMenuItem(): ElementFinder {
    return this.contextMenu.all(by.tagName('button')).get(1);
  }

  get fineTuneDialog(): ElementFinder {
    return element(by.tagName('idlp-fine-tune-dialog'));
  }

  get fineTuneModelName(): ElementFinder {
    return this.fineTuneDialog.element(by.tagName('input[formcontrolname="name"]'));
  }

  get fineTuneModelEpochs(): ElementFinder {
    return this.fineTuneDialog.element(by.tagName('input[formcontrolname="epochs"]'));
  }

  get fineTuneModelBuild(): ElementFinder {
    return this.fineTuneDialog.element(by.tagName('input[formcontrolname="build"]'));
  }

  get fineTuneModelBuildItems(): ElementArrayFinder {
    return element(by.css('.mat-autocomplete-panel')).all(by.tagName('mat-option'));
  }

  get fineTuneDialogAdvancedCheckbox(): ElementFinder {
    return this.fineTuneDialog.element(by.tagName('mat-checkbox[formcontrolname="advanced"]'));
  }

  get fineTuneDialogSubmitButton(): ElementFinder {
    return this.fineTuneDialog.all(by.tagName('button')).first();
  }

  get inProgressRow(): ElementFinder {
    return element(by.css('.in-progress-cell'));
  }

  get cvatHeader(): ElementFinder {
    return element(by.css('.cvat-header'));
  }

  get cvatBody(): ElementFinder {
    return element(by.tagName('body'));
  }

  get cvatUsername(): ElementFinder {
    return element(by.id('username'));
  }

  get cvatPassword(): ElementFinder {
    return element(by.id('password'));
  }

  get cvatSignInButton(): ElementFinder {
    return element(by.css('[type=submit]'));
  }

  get cvatActionButton(): ElementFinder {
    return element(by.cssContainingText('span', 'Actions')).element(by.xpath('..'));
  }

  get cvatUploadAnnotationsButton(): ElementFinder {
    return element(by.css('[title="Upload annotations"]'));
  }

  get cvatUploadCocoAnnotationButton(): ElementFinder {
    return element(by.cssContainingText('span', 'COCO 1.0')).element(by.xpath('..'));
  }

  switchBrowserTab(tabIndex?: number): void {
    browser.getAllWindowHandles()
      .then((handles: string[]) => {
        browser.switchTo().window(handles[tabIndex ?? 0]);
      });
  }

  hasClass(el: any, cls: string): boolean {
    return el.getAttribute('class')
      .then((classes) => classes.split(' ').includes(cls));
  };
}
