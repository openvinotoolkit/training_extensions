/**
 * @overview
 * @copyright (c) JSC Intel A/O
 */

import * as path from 'path';
import {browser, by, element, protractor} from 'protractor';
import {AppPage} from './app.po';

const testModelName = `test-model-${Date.now()}`;

describe('OpenVINO Training Extensions', () => {
  let page: AppPage;

  beforeEach(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 60000;
    browser.waitForAngularEnabled(false);
    browser.driver.manage().window().maximize();
    page = new AppPage();
  });

  it('should navigate to CVAT and sign in with django/django credentials', () => {
    const until = protractor.ExpectedConditions;
    browser.get(`${browser.params.cvat.url}/auth/login`);
    browser.wait(until.presenceOf(page.cvatUsername), 60000, 'CVAT not respond');
    expect(page.cvatUsername.isPresent()).toBeTruthy();
    expect(page.cvatPassword.isPresent()).toBeTruthy();
    expect(page.cvatSignInButton.isPresent()).toBeTruthy();
    page.cvatUsername.sendKeys(browser.params.cvat.credentials.username);
    page.cvatPassword.sendKeys(browser.params.cvat.credentials.password);
    page.cvatSignInButton.click();
    browser.wait(until.presenceOf(page.cvatHeader), 60000, 'Authorization takes too long');
    browser.sleep(500);
  });

  it('should display application title', () => {
    browser.get(browser.params.baseUrl);
    browser.sleep(1000);
    expect(page.toolbarTitle).toEqual('OpenVINO Training Extensions');
  });

  it('should navigate "Face Detection" problem', () => {
    page.faceDetectionProblem.click();
    browser.sleep(1000);
    expect(page.faceDetectionProblemTitle).toEqual('Face Detection');
  });

  it('should navigate "Face Detection" problem assets', () => {
    page.assetsNavigationItem.click();
    browser.sleep(1000);
    expect(page.assetCardItems.count()).toBeGreaterThan(0);
  });

  it('should push assets to CVAT and disable push buttons', () => {
    const until = protractor.ExpectedConditions;

    expect(page.hasClass(page.firstAssetCardItemPushButton.element(by.tagName('div')), 'disabled')).toBeFalsy();
    expect(page.hasClass(page.firstAssetCardItemPullButton.element(by.tagName('div')), 'disabled')).toBeTruthy();
    expect(page.hasClass(page.secondAssetCardItemPushButton.element(by.tagName('div')), 'disabled')).toBeFalsy();
    expect(page.hasClass(page.secondAssetCardItemPullButton.element(by.tagName('div')), 'disabled')).toBeTruthy();

    page.firstAssetCardItemPushButton.click();
    expect(page.firstAssetCardItemLoader.isPresent()).toBeTruthy();

    browser.sleep(500);

    page.secondAssetCardItemPushButton.click();
    expect(page.secondAssetCardItemLoader.isPresent()).toBeTruthy();

    browser.wait(until.stalenessOf(page.firstAssetCardItemLoader), 60000, 'First asset push takes too long');
    browser.wait(until.stalenessOf(page.secondAssetCardItemLoader), 60000, 'Second asset push takes too long');

    expect(page.hasClass(page.firstAssetCardItemPushButton.element(by.tagName('div')), 'disabled')).toBeTruthy();
    expect(page.hasClass(page.firstAssetCardItemPullButton.element(by.tagName('div')), 'disabled')).toBeFalsy();
    expect(page.hasClass(page.secondAssetCardItemPushButton.element(by.tagName('div')), 'disabled')).toBeTruthy();
    expect(page.hasClass(page.secondAssetCardItemPullButton.element(by.tagName('div')), 'disabled')).toBeFalsy();
  });

  it('should navigate to CVAT and upload annotation for the first asset', () => {
    const meetingTestAnnotation = 'data/meeting_test_annotation.json';
    const firstAssetAnnotationPath = path.resolve(__dirname, meetingTestAnnotation);

    page.firstAssetCardItemActionWrapper.click();
    browser.sleep(1000);
    page.switchBrowserTab(1);

    expect(page.cvatActionButton.isPresent()).toBeTruthy();
    page.cvatActionButton.click();
    browser.sleep(1000);

    expect(page.cvatUploadAnnotationsButton.isPresent()).toBeTruthy();
    page.cvatUploadAnnotationsButton.click();
    browser.sleep(1000);

    page.cvatBody.click();
    browser.sleep(500);

    page.cvatActionButton.click();
    browser.sleep(1000);

    page.cvatUploadAnnotationsButton.click();
    browser.sleep(1000);

    expect(page.cvatUploadCocoAnnotationButton.isPresent()).toBeTruthy();
    page.cvatUploadCocoAnnotationButton.click();
    browser.sleep(1000);

    const fileElem = element(by.css('input[type="file"]'));
    fileElem.sendKeys(firstAssetAnnotationPath);
    browser.actions().sendKeys(protractor.Key.ENTER).perform();
    browser.sleep(5000);

    browser.driver.close();
    page.switchBrowserTab();
  });

  it('should navigate to CVAT and upload annotation for the second asset', () => {
    const groupTestAnnotation = 'data/group_test_annotation.json';
    const secondAssetAnnotationPath = path.resolve(__dirname, groupTestAnnotation);

    page.secondAssetCardItemActionWrapper.click();
    browser.sleep(1000);
    page.switchBrowserTab(1);

    expect(page.cvatActionButton.isPresent()).toBeTruthy();
    page.cvatActionButton.click();
    browser.sleep(1000);

    expect(page.cvatUploadAnnotationsButton.isPresent()).toBeTruthy();
    page.cvatUploadAnnotationsButton.click();
    browser.sleep(1000);

    page.cvatBody.click();
    browser.sleep(500);

    page.cvatActionButton.click();
    browser.sleep(1000);

    page.cvatUploadAnnotationsButton.click();
    browser.sleep(1000);

    expect(page.cvatUploadCocoAnnotationButton.isPresent()).toBeTruthy();
    page.cvatUploadCocoAnnotationButton.click();
    browser.sleep(1000);

    const fileElem = element(by.css('input[type="file"]'));
    fileElem.sendKeys(secondAssetAnnotationPath);
    browser.actions().sendKeys(protractor.Key.ENTER).perform();
    browser.sleep(5000);

    browser.driver.close();
    page.switchBrowserTab();
  });

  it('should make initial pull for the first asset from CVAT', () => {
    const until = protractor.ExpectedConditions;

    expect(page.hasClass(page.firstAssetCardItemPullButton.element(by.tagName('div')), 'initial')).toBeTruthy();
    expect(page.hasClass(page.secondAssetCardItemPullButton.element(by.tagName('div')), 'initial')).toBeTruthy();

    page.firstAssetCardItemPullButton.click();
    expect(page.firstAssetCardItemLoader.isPresent()).toBeTruthy();

    page.secondAssetCardItemPullButton.click();
    expect(page.secondAssetCardItemLoader.isPresent()).toBeTruthy();

    browser.wait(until.stalenessOf(page.firstAssetCardItemLoader), 60000, 'First asset pull takes too long');
    browser.wait(until.stalenessOf(page.secondAssetCardItemLoader), 60000, 'Second asset pull takes too long');

    expect(page.hasClass(page.firstAssetCardItemPullButton.element(by.tagName('div')), 'initial')).toBeFalsy();
    expect(page.hasClass(page.secondAssetCardItemPullButton.element(by.tagName('div')), 'initial')).toBeFalsy();
  });

  it('should select assets for train and val and create a new build', () => {
    expect(page.createBuildButton.getAttribute('disabled')).toEqual('true');
    page.firstAssetCardItemTrainCheckbox.click();
    page.secondAssetCardItemValCheckbox.click();
    browser.sleep(1000);

    expect(page.createBuildButton.getAttribute('disabled')).toBeNull();
    page.createBuildButton.click();
    browser.sleep(1000);
  });

  it('should navigate to info page and select a new build', () => {
    page.infoNavigationItem.click();
    browser.sleep(1000);

    expect(page.buildAutocomplete.isPresent()).toBeTruthy();
    page.buildAutocomplete.click();
    browser.sleep(500);

    expect(page.buildsAutocompleteItems.isPresent()).toBeTruthy();
    page.buildsAutocompleteItems.get(1).click();
    browser.sleep(500);
  });

  it('should select model and click fine tune', () => {
    browser.executeScript('document.getElementsByClassName(\'scroll-container\')[0].scrollTo(0, document.getElementsByClassName(\'scroll-container\')[0].scrollHeight)');
    browser.sleep(500);
    page.fineTuneModel.click();
    browser.sleep(1000);
    expect(page.contextMenu.isPresent()).toBeTruthy();
    expect(page.fineTuneMenuItem.isPresent()).toBeTruthy();
    page.fineTuneMenuItem.click();
    browser.sleep(500);
    expect(page.fineTuneDialog.isPresent()).toBeTruthy();
  });

  it('should populate fine tune dialog fields and submit', () => {
    expect(page.fineTuneModelName.isPresent()).toBeTruthy();
    expect(page.fineTuneModelBuild.isPresent()).toBeTruthy();
    expect(page.fineTuneModelEpochs.isPresent()).toBeTruthy();

    page.fineTuneModelName.clear();
    page.fineTuneModelName.sendKeys(testModelName);
    browser.sleep(500);

    page.fineTuneModelBuild.click();
    browser.sleep(500);
    expect(page.fineTuneModelBuildItems.isPresent()).toBeTruthy();
    page.fineTuneModelBuildItems.first().click();

    page.fineTuneModelEpochs.clear();
    page.fineTuneModelEpochs.sendKeys(1);
    browser.sleep(500);

    page.fineTuneDialogSubmitButton.click();
  });

  it('should wait for training complete', () => {
    const until = protractor.ExpectedConditions;
    browser.refresh();
    browser.wait(until.presenceOf(page.inProgressRow), 60000, 'Seems test model training has not been started');
    page.buildAutocomplete.click();
    browser.sleep(500);
    expect(page.buildsAutocompleteItems.isPresent()).toBeTruthy();
    page.buildsAutocompleteItems.get(1).click();
    browser.sleep(500);
    browser.executeScript('document.getElementsByClassName(\'scroll-container\')[0].scrollTo(0, document.getElementsByClassName(\'scroll-container\')[0].scrollHeight)');
    expect(page.inProgressRow.isPresent()).toBeTruthy();
    browser.wait(until.stalenessOf(page.inProgressRow), 600000, 'Test model training taking too long');
    expect(page.inProgressRow.isPresent()).toBeFalsy();
    browser.sleep(1000);
  });

  it('should navigate to file browser and check if model exist', () => {
    browser.get('http://localhost:8003/files/idlp/problem/Face_Detection/models/');
    browser.sleep(1000);
    browser.executeScript('window.scrollTo(0, document.body.scrollHeight)');
    expect(element(by.tagName(`div[aria-label="${testModelName}"]`)).isPresent()).toBeTruthy();
    browser.sleep(1000);
  });

  it('should navigate to model content and check if model\'s artifacts exists', () => {
    browser.actions().doubleClick(element(by.tagName(`div[aria-label="${testModelName}"]`))).perform();
    browser.sleep(1000);
    expect(element(by.tagName('div[aria-label="model.yml"]')).isPresent()).toBeTruthy();
    expect(element(by.tagName('div[aria-label="snapshot.pth"]')).isPresent()).toBeTruthy();
    expect(element(by.tagName('div[aria-label="config.py"]')).isPresent()).toBeTruthy();
    browser.sleep(1000);
  });

  // afterEach(async () => {
  //   const logs = await browser.manage().logs().get(logging.Type.BROWSER);
  //   expect(logs).not.toContain(jasmine.objectContaining({
  //     level: logging.Level.SEVERE,
  //   } as logging.Entry));
  // });
});
