# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
Feature: Export Dataset
  As a user of the application
  I want to export datasets in various formats
  So that I can use the data in other applications

  @export @detection
  Scenario Outline: Export detection project dataset
    Given A detection project "grapes" with labels ["Chardonnay", "Sauvignon Blanc", "Cabernet Franc"] exists
    And the project contains the following image distribution:
      | Label           | Training | Validation | Testing |
      | Chardonnay      | 6        | 2          | 2       |
      | Sauvignon Blanc | 6        | 2          | 2       |
      | Cabernet Franc  | 6        | 2          | 2       |
    And the project contains the following video frame distribution:
      | Label           | Training | Validation | Testing |
      | Chardonnay      | 2        | 1          | 1       |
    When I export the project dataset in <export format> format with filters=<filters>
    Then the staged dataset archive <archive name> should exist
    And the staged dataset with name=<archive name> has <image count> images
    And the staged dataset with name=<archive name> has <frame count> video frames

    Examples:
      | export format | archive name     | filters                                                    | image count | frame count |
      | YOLO          | test_project-yolo-dataset.zip | { }                                                        | 34          | 0           |
      | YOLO          | test_project-yolo-dataset.zip | { "labels": ["Chardonnay"], "include_unannotated": false } | 14          | 0           |
      | GETI          | test_project-geti-dataset.zip | { }                                                        | 30          | 4           |
      | GETI          | test_project-geti-dataset.zip | { "labels": ["Chardonnay"], "include_unannotated": false } | 10          | 4           |
      | COCO          | test_project-coco-dataset.zip | { }                                                        | 34          | 0           |
      | COCO          | test_project-coco-dataset.zip | { "labels": ["Chardonnay"], "include_unannotated": false } | 14          | 0           |

  @export @classification
  Scenario Outline: Export classification project dataset
    Given A classification project "animals" with labels ["cat", "dog"] exists
    And the project contains the following image distribution:
      | Label | Training | Validation | Testing |
      | cat   | 5        | 5          | 5       |
      | dog   | 5        | 5          | 5       |
    And the project contains the following video frame distribution:
      | Label | Training | Validation | Testing |
      | cat   | 3        | 1          | 1       |
      | dog   | 3        | 1          | 1       |
    When I export the project dataset in <export format> format with filters=<filters>
    Then the staged dataset archive <archive name> should exist
    And the staged dataset with name=<archive name> has <image count> images
    And the staged dataset with name=<archive name> has <frame count> video frames

    Examples:
      | export format | archive name     | filters                                             | image count | frame count |
      | GETI          | test_project-geti-dataset.zip | { }                                                 | 30          | 10          |
      | GETI          | test_project-geti-dataset.zip | { "labels": ["cat"], "include_unannotated": false } | 15          | 5           |
      | VOC           | test_project-voc-dataset.zip  | { }                                                 | 40          | 0           |
      | VOC           | test_project-voc-dataset.zip  | { "labels": ["cat"], "include_unannotated": false } | 20          | 0           |

  @export @segmentation
  Scenario Outline: Export segmentation project dataset
    Given An instance_segmentation project "traffic" with labels ["car", "person"] exists
    And the project contains the following image distribution:
      | Label  | Training | Validation | Testing |
      | car    | 7        | 5          | 3       |
      | person | 7        | 5          | 3       |
    And the project contains the following video frame distribution:
      | Label  | Training | Validation | Testing |
      | car    | 1        | 1          | 1       |
      | person | 1        | 1          | 1       |
    When I export the project dataset in <export format> format with filters=<filters>
    Then the staged dataset archive <archive name> should exist
    And the staged dataset with name=<archive name> has <image count> images
    And the staged dataset with name=<archive name> has <frame count> video frames

    Examples:
      | export format | archive name     | filters                       | image count | frame count |
      | GETI          | test_project-geti-dataset.zip | { }                           | 30           | 6          |
      | GETI          | test_project-geti-dataset.zip | { "labels": ["person"] }      | 30           | 6          |
      | COCO          | test_project-coco-dataset.zip | { }                           | 36           | 0          |
      | COCO          | test_project-coco-dataset.zip | { "labels": ["person"] }      | 36           | 0          |
