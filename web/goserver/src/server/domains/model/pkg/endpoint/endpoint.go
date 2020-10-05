package endpoint

import (
	"context"

	"server/domains/model/pkg/service"
	kitendpoint "server/kit/endpoint"
)

type Endpoints struct {
	CreateFromGeneric kitendpoint.Endpoint
	Evaluate          kitendpoint.Endpoint
	FineTune          kitendpoint.Endpoint
	List              kitendpoint.Endpoint
	UpdateFromLocal   kitendpoint.Endpoint
}

func New(s service.ModelService, mdw map[string][]kitendpoint.Middleware) Endpoints {
	eps := Endpoints{
		CreateFromGeneric: MakeCreateFromGenericEndpoint(s),
		Evaluate:          MakeEvaluateEndpoint(s),
		FineTune:          MakeFineTuneEndpoint(s),
		List:              MakeListEndpoint(s),
		UpdateFromLocal:   MakeUpdateFromLocalEnpoint(s),
	}
	return eps
}

func MakeCreateFromGenericEndpoint(s service.ModelService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		req := request.(service.CreateFromGenericRequest)
		return s.CreateFromGeneric(ctx, req)
	}
}

func MakeEvaluateEndpoint(s service.ModelService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		req := request.(service.EvaluateRequest)
		return s.Evaluate(ctx, req)
	}
}

func MakeFineTuneEndpoint(s service.ModelService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		req := request.(service.FineTuneRequestData)
		return s.FineTune(ctx, req)

	}
}

func MakeListEndpoint(s service.ModelService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		req := request.(service.ListRequestData)
		return s.List(ctx, req)
	}
}

func MakeUpdateFromLocalEnpoint(s service.ModelService) kitendpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan kitendpoint.Response {
		return s.UpdateFromLocal(ctx, request.(service.UpdateFromLocalRequestData))
	}
}
