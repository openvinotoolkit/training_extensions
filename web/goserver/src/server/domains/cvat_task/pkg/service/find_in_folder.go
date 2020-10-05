package service

import (
	"context"
	"strings"

	"go.mongodb.org/mongo-driver/bson/primitive"

	assetFind "server/db/pkg/handler/asset/find"
	buildFindOne "server/db/pkg/handler/build/find_one"
	cvatTaskFind "server/db/pkg/handler/cvat_task/find"
	cvatTaskInsertOne "server/db/pkg/handler/cvat_task/insert_one"
	problemFindOne "server/db/pkg/handler/problem/find_one"
	t "server/db/pkg/types"
	buildStatus "server/db/pkg/types/build/status"
	typeAsset "server/db/pkg/types/type/asset"
	buildCreateEmpty "server/domains/build/pkg/handler/create_empty"
)

type FindInFolderRequestData struct {
	ProblemId primitive.ObjectID `bson:"problemId" json:"problemId"`
	Root      string             `bson:"root" json:"root"`
	Page      int64              `bson:"page" json:"page"`
	Size      int64              `bson:"size" json:"size"`
}

type BuildSplit struct {
	Train int `bson:"train" json:"train"`
	Test  int `bson:"test" json:"test"`
	Val   int `bson:"val" json:"val"`
}

type CvatTaskAndAsset struct {
	Id           primitive.ObjectID `bson:"id" json:"id"`
	CvatId       int                `bson:"cvatId" json:"cvatId"`
	ProblemId    primitive.ObjectID `bson:"problemId" json:"problemId"`
	ParentFolder string             `bson:"parentFolder" json:"parentFolder"`
	Name         string             `bson:"name" json:"name"`
	Type         string             `bson:"type" json:"type"`
	Status       string             `bson:"status" json:"status"`
	Url          string             `bson:"url" json:"url"`
	Progress     t.CvatTaskProgress `bson:"progress" json:"progress"`
	BuildSplit   BuildSplit         `bson:"buildSplit" json:"buildSplit"`
}

type FindInFolderResponseData struct {
	Total int64              `bson:"total" json:"total"`
	Items []CvatTaskAndAsset `bson:"items" json:"items"`
}

func (s *basicCvatTaskService) FindInFolder(ctx context.Context, req FindInFolderRequestData) (result FindInFolderResponseData) {
	assetFindResp := <-assetFind.Send(
		ctx,
		s.Conn,
		assetFind.RequestData{
			ParentFolder: req.Root,
			Page:         req.Page,
			Size:         req.Size,
		},
	)
	assets := assetFindResp.Data.(assetFind.ResponseData).Items

	buildFindOneResp := <-buildFindOne.Send(
		ctx,
		s.Conn,
		buildFindOne.RequestData{
			ProblemId: req.ProblemId,
			Status:    buildStatus.Tmp,
		},
	)
	if primitive.ObjectID.IsZero(buildFindOneResp.Data.(buildFindOne.ResponseData).Id) {
		buildFindOneResp = <-buildCreateEmpty.Send(
			ctx,
			s.Conn,
			buildCreateEmpty.RequestData{
				ProblemId: req.ProblemId,
			},
		)
	}
	tmpBuild := buildFindOneResp.Data.(buildFindOne.ResponseData)
	result.Total = assetFindResp.Data.(assetFind.ResponseData).Total

	var assetIds []primitive.ObjectID
	for _, asset := range assets {
		assetIds = append(assetIds, asset.Id)
	}
	// Get cvat tasks associated with assets
	cvatTaskFindResp := <-cvatTaskFind.Send(
		ctx,
		s.Conn,
		cvatTaskFind.RequestData{
			ProblemId: req.ProblemId,
			AssetIds:  assetIds,
		},
	)
	var cvatTasks []t.CvatTask
	if cvatTaskFindResp.Data != nil {
		cvatTasks = cvatTaskFindResp.Data.(cvatTaskFind.ResponseData).Items
	}
	problemResp := <-problemFindOne.Send(
		ctx,
		s.Conn,
		problemFindOne.RequestData{
			Id: req.ProblemId,
		},
	)
	problem := problemResp.Data.(problemFindOne.ResponseData)

	// Look for assets that don't have cvat task
	var assetsWithoutCvatTask []t.Asset
AssetLoop1:
	for _, asset := range assets {
		for _, cvatTask := range cvatTasks {
			if asset.Id == cvatTask.AssetId {
				continue AssetLoop1
			}
		}
		assetsWithoutCvatTask = append(assetsWithoutCvatTask, asset)
	}

	// If asset without cvat task exists, create cvat tasks for those assets
	for _, asset := range assetsWithoutCvatTask {
		assetPath := strings.Join([]string{asset.ParentFolder, asset.Name}, "/")
		if asset.Type == typeAsset.Folder {
			go s.FindInFolder(ctx, FindInFolderRequestData{
				ProblemId: req.ProblemId,
				Root:      assetPath,
			})
		}
		cvatTaskInsertResp := <-cvatTaskInsertOne.Send(
			ctx,
			s.Conn,
			cvatTaskInsertOne.RequestData{
				ProblemId: problem.Id,
				AssetId:   asset.Id,
				AssetPath: assetPath,
				Status:    "initial",
				Params: t.CVATParams{
					Name:         assetPath,
					ImageQuality: 70,
					ZOrder:       false,
					BugTracker:   "",
					SegmentSize:  5000,
					Labels:       problem.Labels,
				},
				Progress: t.CvatTaskProgress{
					Total:      0,
					Done:       0,
					Percentage: 0.0,
				},
			},
		)
		newCvatTask := cvatTaskInsertResp.Data.(t.CvatTask)
		cvatTasks = append(cvatTasks, newCvatTask)
	}

	for _, asset := range assets {
		cvatTask := findCvatTaskByAssetId(cvatTasks, asset.Id)
		buildSplit := findBuildAssetSplit(tmpBuild, asset)
		result.Items = append(result.Items, getCvatTaskAndAsset(cvatTask, asset, buildSplit))
	}
	return result
}

func getCvatTaskAndAsset(cvatTask t.CvatTask, asset t.Asset, buildSplit BuildSplit) CvatTaskAndAsset {
	return CvatTaskAndAsset{
		Id:           cvatTask.Id,
		ProblemId:    cvatTask.ProblemId,
		ParentFolder: asset.ParentFolder,
		Name:         asset.Name,
		Type:         asset.Type,
		Status:       cvatTask.Status,
		CvatId:       cvatTask.Annotation.Id,
		Progress:     cvatTask.Progress,
		BuildSplit:   buildSplit,
		Url:          cvatTask.Annotation.Url,
	}
}

func findCvatTaskByAssetId(cvatTasks []t.CvatTask, assetId primitive.ObjectID) t.CvatTask {
	for _, cvatTask := range cvatTasks {
		if cvatTask.AssetId == assetId {
			return cvatTask
		}
	}
	return t.CvatTask{}
}

func findBuildAssetSplit(build t.Build, asset t.Asset) (result BuildSplit) {
	path := strings.Split(asset.ParentFolder, "/")
	buildAssetSplit := build.Split[path[0]]
	for _, folder := range path[1:] {
		buildAssetSplit = buildAssetSplit.Children[folder]
	}
	buildAssetSplit = buildAssetSplit.Children[asset.Name]
	return BuildSplit{
		Train: buildAssetSplit.Train,
		Test:  buildAssetSplit.Test,
		Val:   buildAssetSplit.Val,
	}
}
