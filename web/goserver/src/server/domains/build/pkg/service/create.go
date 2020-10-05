package service

import (
	"context"
	"fmt"
	"log"
	"os"
	fp "path/filepath"
	"strconv"
	"strings"

	"go.mongodb.org/mongo-driver/bson/primitive"

	buildFindOne "server/db/pkg/handler/build/find_one"
	buildInsertOne "server/db/pkg/handler/build/insert_one"
	cvatTaskFind "server/db/pkg/handler/cvat_task/find"
	problemFindOne "server/db/pkg/handler/problem/find_one"
	t "server/db/pkg/types"
	buildStatus "server/db/pkg/types/build/status"
	"server/kit/utils/basic/arrays"
	ufiles "server/kit/utils/basic/files"
)

type CreateRequestData struct {
	ProblemId primitive.ObjectID `bson:"problemId" json:"problemId"`
	Name      string             `bson:"name" json:"name"`
}

func (s *basicBuildService) Create(_ context.Context, req CreateRequestData) {
	problem := s.getProblem(req.ProblemId)
	tmpBuild := s.getTmpBuild(problem.Id)
	buildFolderName := getBuildFolderName(req.Name)
	buildFolderPath := createBuildFolder(problem.Dir, buildFolderName)
	tmpFolderPath := fmt.Sprintf("%s/_builds/%s", problem.Dir, tmpBuild.Folder)
	annotationIdsList := s.getAnnotationIdsInBuild(tmpBuild)
	copyAnnotationsFromTmpToBuildFolder(tmpFolderPath, buildFolderPath, annotationIdsList)
	s.createNewBuild(tmpBuild, req.Name, buildFolderName)
}

func getBuildFolderName(name string) string {
	folder := strings.ReplaceAll(name, " ", "_")
	folder = strings.ReplaceAll(folder, "/", "_")
	return folder
}

func (s *basicBuildService) createNewBuild(tmpBuild t.Build, name, folder string) {
	<-buildInsertOne.Send(
		context.TODO(),
		s.Conn,
		buildInsertOne.RequestData{
			ProblemId: tmpBuild.ProblemId,
			Folder:    folder,
			Name:      name,
			Split:     tmpBuild.Split,
			Status:    buildStatus.Ready,
		},
	)
}

func (s *basicBuildService) getTmpBuild(problemId primitive.ObjectID) t.Build {
	buildFindOneResp := <-buildFindOne.Send(
		context.TODO(),
		s.Conn,
		buildFindOne.RequestData{
			ProblemId: problemId,
			Status:    buildStatus.Tmp,
		},
	)
	return buildFindOneResp.Data.(buildFindOne.ResponseData)
}

func (s *basicBuildService) getProblem(problemId primitive.ObjectID) t.Problem {
	problemFindOneResp := <-problemFindOne.Send(
		context.TODO(),
		s.Conn,
		problemFindOne.RequestData{Id: problemId},
	)
	return problemFindOneResp.Data.(problemFindOne.ResponseData)
}

func copyAnnotationsFromTmpToBuildFolder(src, dst string, idsList []int) {
	err := fp.Walk(src, func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}
		if isAnnotationInList(idsList, info.Name()) {
			from := fp.Join(src, info.Name())
			to := fp.Join(dst, info.Name())
			_, err := ufiles.Copy(from, to)
			if err != nil {
				log.Println("domains.build.pkg.service.create.copyAnnotationsFromTmpToBuildFolder.ufiles.Copy(from, to)", err)
			}
		}
		return nil
	})
	if err != nil {
		log.Println("domains.build.pkg.service.create.copyAnnotationsFromTmpToBuildFolder.fp.Walk", err)
	}
}
func isAnnotationInList(idsList []int, name string) bool {
	annFileId, err := strconv.Atoi(strings.Split(name, ".")[0])
	if err != nil {
		log.Println("domains.build.pkg.service.create.isAnnotationInList.strconv.Atoi(strings.Split(name, \".\")[0])", err)
	}
	return arrays.ContainsInt(idsList, annFileId)
}

func createBuildFolder(problemDir, name string) string {
	path := fmt.Sprintf("%s/_builds/%s", problemDir, name)
	if err := os.MkdirAll(path, 0777); err != nil {
		log.Println("domains.build.pkg.service.create.createBuildFolder.os.MkdirAll(path, 0777)", err)
	}
	return path
}

func (s *basicBuildService) getAnnotationIdsInBuild(build t.Build) []int {
	var result []int
	assetIdsList := getAssetIdsIncludedInBuild(build.Split["."].Children)
	cvatTaskFindResp := <-cvatTaskFind.Send(
		context.TODO(),
		s.Conn,
		cvatTaskFind.RequestData{
			ProblemId: build.ProblemId,
			AssetIds:  assetIdsList,
		},
	)
	cvatTasks := cvatTaskFindResp.Data.(cvatTaskFind.ResponseData).Items
	for _, cvatTask := range cvatTasks {
		result = append(result, cvatTask.Annotation.Id)
	}
	return result

}

func getAssetIdsIncludedInBuild(buildSplit map[string]t.BuildAssetsSplit) []primitive.ObjectID {
	var result []primitive.ObjectID
	for _, child := range buildSplit {
		if len(child.Children) == 0 && (child.Test+child.Val+child.Train) > 0 {
			result = append(result, child.AssetId)
		} else {
			childAssetIdsList := getAssetIdsIncludedInBuild(child.Children)
			result = append(result, childAssetIdsList...)
		}
	}
	return result
}
