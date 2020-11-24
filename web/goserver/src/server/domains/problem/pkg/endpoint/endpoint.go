package endpoint

import (
	"context"

	"server/domains/problem/pkg/service"
	kitendpoint "server/kit/endpoint"
)

type Endpoints struct {
	Create          kitendpoint.Endpoint
	Delete          kitendpoint.Endpoint
	Details         kitendpoint.Endpoint
	List            kitendpoint.Endpoint
	UpdateFromLocal kitendpoint.Endpoint
}

// New returns a Endpoints struct that wraps the provided service, and wires in all of the
// expected endpoint middlewares
func New(s service.ProblemService, mdw map[string][]kitendpoint.Middleware) Endpoints {
	eps := Endpoints{
		Create:          MakeCreateEndpoint(s),
		Delete:          MakeDeleteEndpoint(s),
		Details:         MakeDetailsEndpoint(s),
		List:            MakeListEndpoint(s),
		UpdateFromLocal: MakeUpdateFromLocalEndpoint(s),
	}

	// for _, m := range mdw["WebSocket"] {
	// 	eps.Details = m(eps.Details)
	// }
	return eps
}

func MakeCreateEndpoint(s service.ProblemService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		req := request.(service.CreateRequestData)
		responseChan := make(chan kitendpoint.Response)
		go s.Create(ctx, req, responseChan)
		return responseChan
	}
}

func MakeDeleteEndpoint(s service.ProblemService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		req := request.(service.DeleteRequestData)
		responseChan := make(chan kitendpoint.Response)
		go s.Delete(ctx, req, responseChan)
		return responseChan
	}
}

func MakeDetailsEndpoint(s service.ProblemService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		req := request.(service.DetailsRequestData)
		responseChan := make(chan kitendpoint.Response)
		go s.Details(ctx, req, responseChan)
		return responseChan
	}
}

func MakeListEndpoint(s service.ProblemService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		req := request.(service.ListRequestData)
		responseChan := make(chan kitendpoint.Response)
		go s.List(ctx, req, responseChan)
		return responseChan
	}
}

func MakeUpdateFromLocalEndpoint(s service.ProblemService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		req := request.(service.UpdateFromLocalRequestData)
		responseChan := make(chan kitendpoint.Response)
		go s.UpdateFromLocal(ctx, req, responseChan)
		return responseChan
	}
}
