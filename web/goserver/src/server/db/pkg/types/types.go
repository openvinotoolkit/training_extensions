package types

import (
	"go.mongodb.org/mongo-driver/bson/primitive"
)

type BaseList struct {
	Total int64 `json:"total"`
}

type OneRequest struct {
	Id primitive.ObjectID `bson:"_id" json:"id"`
}

type Asset struct {
	Id           primitive.ObjectID `bson:"_id" json:"id"`
	ParentFolder string             `bson:"parentFolder" json:"parentFolder"`
	Name         string             `bson:"name" json:"name"`
	Type         string             `bson:"type" json:"type"`
	CvatDataPath string             `bson:"cvatDataPath" json:"cvatDataPath"`
}

type AssetFindResponse struct {
	BaseList
	Items []Asset `bson:"items" json:"items"`
}

type BuildAssetsSplit struct {
	AssetId              primitive.ObjectID          `bson:"assetId" json:"assetId"`
	CvatTaskAnnotationId int                         `bson:"cvatTaskAnnotationId" json:"cvatCvatAnnotationId"`
	Children             map[string]BuildAssetsSplit `bson:"children" json:"children"`
	Test                 int                         `bson:"test" json:"test"`
	Train                int                         `bson:"train" json:"train"`
	Val                  int                         `bson:"val" json:"val"`
}

type Build struct {
	Id        primitive.ObjectID          `bson:"_id" json:"id"`
	ProblemId primitive.ObjectID          `bson:"problemId" json:"problemId"`
	Name      string                      `bson:"name" json:"name"`
	Split     map[string]BuildAssetsSplit `bson:"split" json:"split"`
	Status    string                      `bson:"status" json:"status"`
	Folder    string                      `bson:"folder" json:"folder"`
}

type BuildFindResponse struct {
	BaseList
	Items []Build `bson:"items" json:"items"`
}

type CVATParams struct {
	ImageQuality int                      `bson:"imageQuality" json:"-"`
	ZOrder       bool                     `bson:"zOrder" json:"-"`
	BugTracker   string                   `bson:"bugTracker" json:"-"`
	SegmentSize  int                      `bson:"segmentSize" json:"segment_size"`
	Labels       []map[string]interface{} `bson:"labels" json:"labels"`
	Name         string                   `bson:"name" json:"name"`
}

type CvatTaskProgress struct {
	Total      int64   `bson:"total"      json:"total"`
	Done       int64   `bson:"done"       json:"done"`
	Percentage float64 `bson:"percentage" json:"percentage"`
}

type CvatTaskCreateTaskStatus struct {
	State   string `bson:"state" json:"state"`
	Message string `bson:"message" json:"message"`
}

type CvatJob struct {
	StartFrame int64  `bson:"start_frame" json:"start_frame"`
	StopFrame  int64  `bson:"stop_frame" json:"stop_frame"`
	Url        string `bson:"url" json:"url"`
	Id         int64  `bson:"id" json:"id"`
	Status     string `bson:"status" json:"status"`
}

type CvatAnnotation struct {
	Id     int       `bson:"id" json:"id"`
	Status string    `bson:"status" json:"status"`
	Url    string    `bson:"url" json:"url"`
	Jobs   []CvatJob `bson:"job" json:"job"`
}

type CvatTask struct {
	Annotation       CvatAnnotation           `bson:"annotation" json:"annotation"`
	AssetId          primitive.ObjectID       `bson:"assetId" json:"assetId"`
	AssetPath        string                   `bson:"assetPath" json:"assetPath"`
	Status           string                   `bson:"status" json:"status"`
	CreateTaskStatus CvatTaskCreateTaskStatus `bson:"createTaskStatus" json:"createTaskStatus"`
	ProblemId        primitive.ObjectID       `bson:"problemId" json:"problemId"`
	Id               primitive.ObjectID       `bson:"_id" json:"id"`
	Params           CVATParams               `bson:"params" json:"params"`
	Progress         CvatTaskProgress         `bson:"progress" json:"progress"`
}

type CvatTaskFindResponse struct {
	BaseList
	Items []CvatTask `bson:"items"`
}

type Scripts struct {
	Train string `bson:"train" json:"train"`
	Eval  string `bson:"eval" json:"eval"`
}

type Model struct {
	ConfigPath        string              `bson:"configPath" json:"configPath"`
	ProblemId         primitive.ObjectID  `bson:"problemId" json:"problemId"`
	Description       string              `bson:"description" json:"description" yaml:"description"`
	Dir               string              `bson:"dir" json:"dir"`
	Dependencies      []Dependency        `bson:"dependencies" json:"dependencies" yaml:"dependencies"`
	Framework         string              `bson:"framework" json:"framework" yaml:"framework"`
	Id                primitive.ObjectID  `bson:"_id" json:"id"`
	Metrics           map[string][]Metric `bson:"metrics,omitempty" json:"metrics,omitempty" yaml:"metrics,omitempty"`
	Name              string              `bson:"name" json:"name" yaml:"name"`
	ParentModelId     primitive.ObjectID  `bson:"parentModelId" json:"parentModelId"`
	Scripts           Scripts             `bson:"scripts" json:"scripts"`
	SnapshotPath      string              `bson:"snapshotPath" json:"snapshotPath"`
	Status            string              `bson:"status" json:"status"`
	TemplatePath      string              `bson:"templatePath" json:"templatePath"`
	TensorBoardLogDir string              `bson:"tensorBoardLogDir" json:"tensorBoardLogDir"`
	TrainingGpuNum    int                 `bson:"trainingGpuNum" json:"trainingGpuNum"`
	TrainingWorkDir   string              `bson:"trainingWorkDir" json:"trainingWorkDir"`
}

type Metric struct {
	DisplayName string  `bson:"displayName" json:"displayName" yaml:"display_name"`
	Key         string  `bson:"key" json:"key" yaml:"key"`
	Value       float64 `bson:"value" json:"value" yaml:"value,omitempty"`
	Unit        string  `bson:"unit" json:"unit" yaml:"unit"`
}

type Dependency struct {
	Sha256      string `yaml:"sha256,omitempty"`
	Size        int    `yaml:"size,omitempty"`
	Source      string `yaml:"source"`
	Destination string `yaml:"destination"`
}

type ModelFindResponse struct {
	BaseList
	Items []Model `bson:"items" json:"items"`
}

// TODO: delete CvatSchema
type Problem struct {
	Class       string                   `bson:"class" json:"class"`
	Description string                   `bson:"description" json:"description"`
	Id          primitive.ObjectID       `bson:"_id" json:"id"`
	ImagesUrls  []string                 `bson:"imagesUrls" json:"imagesUrls" yaml:"imagesUrls"`
	Labels      []map[string]interface{} `bson:"labels" json:"labels"`
	Dir         string                   `bson:"dir" json:"dir"`
	Subtitle    string                   `bson:"subtitle" json:"subtitle"`
	Title       string                   `bson:"title" json:"title"`
	Type        string                   `bson:"type" json:"type"`
	WorkingDir  string                   `bson:"workingDir" json:"workingDir"`
	CvatSchema  string                   `bson:"-" json:"-" yaml:"cvat_schema"`
}

type ProblemFindResponse struct {
	BaseList
	Items []Problem `bson:"items" json:"items"`
}

type Domain struct {
	Problems []Problem `bson:"problems" json:"problems" yaml:"problems"`
	Title    string    `bson:"title" json:"title" yaml:"title"`
}
