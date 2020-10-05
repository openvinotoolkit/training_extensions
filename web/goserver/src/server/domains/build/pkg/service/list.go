package service

import (
	"context"
	"log"

	"go.mongodb.org/mongo-driver/bson/primitive"

	buildFind "server/db/pkg/handler/build/find"
	t "server/db/pkg/types"
	kitendpoint "server/kit/endpoint"
)

type ListRequestData struct {
	Page      int64  `json:"page"`
	Size      int64  `json:"size"`
	ProblemId string `json:"problemId"`
}

func (s *basicBuildService) List(
	_ context.Context,
	req ListRequestData,
) chan kitendpoint.Response {
	returnChan := make(chan kitendpoint.Response)
	go func() {
		defer close(returnChan)
		buildsListData := s.getBuildsListData(req.ProblemId, req.Size, req.Page)
		returnChan <- kitendpoint.Response{
			Data:   buildsListData,
			Err:    nil,
			IsLast: true,
		}
	}()
	return returnChan
}

func (s *basicBuildService) getBuildsListData(problemIdString string, size, page int64) t.BuildFindResponse {
	problemId, err := primitive.ObjectIDFromHex(problemIdString)
	if err != nil {
		log.Println("primitive.ObjectIDFromHex(problemIdString", err)
	}
	buildFindResp := <-buildFind.Send(
		context.TODO(),
		s.Conn,
		buildFind.RequestData{
			Page:      page,
			Size:      size,
			ProblemId: problemId,
		},
	)
	return buildFindResp.Data.(buildFind.ResponseData)
}
